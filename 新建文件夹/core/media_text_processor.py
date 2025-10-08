import hashlib
import os
import pandas as pd
from .base_processor import BaseProcessor
from .utils import (
    UX,
    safe_str_convert,
    normalize_text,
    count_tokens,
    identify_source,
    create_unified_record,
    detect_language_and_get_config,
    ProcessingStatus,
)

class MediaTextProcessor(BaseProcessor):
    """媒体文本处理器 - 两阶段分析"""
    
    async def process_row(self, row_data, output_file_path=None, failed_unit_ids=None):
        """处理单行数据（两阶段模型重构版）"""
        row = row_data[1]
        # 获取媒体文本专用的列映射配置
        media_config = self.config.get('media_text', {})
        COLUMN_MAPPING = media_config.get('column_mapping', {})
        original_id = safe_str_convert(row.get(COLUMN_MAPPING.get("ID", "序号")))
        source = row_data[2]
        
        # 获取失败的单元ID（用于断点续传）
        failed_units = set()
        if failed_unit_ids and isinstance(failed_unit_ids, dict):
            failed_units = set(failed_unit_ids.get(original_id, set()))

        try:
            full_text = safe_str_convert(row.get(COLUMN_MAPPING.get("MEDIA_TEXT", "text"), ''))
            article_title = safe_str_convert(row.get(COLUMN_MAPPING.get('MEDIA_TITLE', "标题"), '无标题'))
            
            if not full_text.strip():
                UX.warn(f"ID {original_id}: 文章内容为空，跳过处理")
                no_relevant_record = create_unified_record(
                    ProcessingStatus.NO_RELEVANT, original_id, source,
                    unit_text="[文章内容为空]",
                    extra_fields={"Article_ID": original_id, "text": full_text}
                )
                return original_id, [no_relevant_record]

            # 检查文本长度限制
            language, config = detect_language_and_get_config(full_text, self.config)
            text_tokens = count_tokens(full_text)
            if text_tokens > config['MAX_SINGLE_TEXT']:
                UX.warn(f"ID {original_id}: 文本过长({text_tokens} tokens)，跳过处理")
                failed_record = create_unified_record(
                    ProcessingStatus.STAGE_1_FAILED,
                    original_id,
                    source,
                    unit_id=original_id
                )
                return original_id, [failed_record]
            
            # 断点续传逻辑：如果有失败单元，从输出文件复用已切分的单元
            if failed_units and output_file_path and os.path.exists(output_file_path):
                UX.info(f"ID {original_id} | 断点续传：复用已提取的{len(failed_units)}个单元")
                try:
                    df_existing = pd.read_excel(output_file_path)
                    existing_units = df_existing[
                        (df_existing[COLUMN_MAPPING["ID"]].astype(str) == original_id) &
                        (df_existing['Unit_ID'].isin(failed_units))
                    ]
                    
                    if not existing_units.empty:
                        retry_records = []

                        for _, unit_row in existing_units.iterrows():
                            unit_id = safe_str_convert(unit_row.get("Unit_ID"))
                            unit_text = safe_str_convert(unit_row.get("Unit_Text", "")).strip()

                            if not unit_text:
                                UX.warn(f"{unit_id} | 单元文本为空，跳过")
                                continue

                            seed_sentence = safe_str_convert(unit_row.get("seed_sentence", ""))
                            expansion_logic = safe_str_convert(unit_row.get("expansion_logic", ""))
                            # 清理旧的错误标记，确保不继承 API_CALL_FAILED
                            raw_speaker = safe_str_convert(unit_row.get("speaker", ""))
                            unit_speaker = raw_speaker if raw_speaker and raw_speaker != 'API_CALL_FAILED' else identify_source(source)

                            analysis_prompt = (
                                self.prompts.UNIT_ANALYSIS
                                .replace("{speaker}", unit_speaker)
                                .replace("{unit_text}", unit_text)
                            )

                            analysis_result = await self.api_service.get_analysis(
                                analysis_prompt,
                                None,
                                language,
                                stage_key="media_text_analysis",
                                context_label=f"{unit_id}:ANALYSIS",
                            )

                            if analysis_result is None:
                                # 失败静默处理，最终汇总会体现
                                retry_records.append(
                                    create_unified_record(
                                        ProcessingStatus.STAGE_2_FAILED,
                                        original_id,
                                        source,
                                        unit_id=unit_id,
                                        unit_text=unit_text,
                                        speaker=unit_speaker,
                                        extra_fields={
                                            "seed_sentence": seed_sentence,
                                            "expansion_logic": expansion_logic,
                                            "Unit_Hash": unit_row.get("Unit_Hash", ""),
                                        },
                                    )
                                )
                                continue

                            norm_text = normalize_text(unit_text)
                            unit_hash = hashlib.sha256(norm_text.encode("utf-8")).hexdigest()

                            retry_records.append(
                                {
                                    "Unit_ID": unit_id,
                                    "Source": source,
                                    "speaker": unit_speaker,
                                    "Unit_Text": unit_text,
                                    "seed_sentence": seed_sentence,
                                    "expansion_logic": expansion_logic,
                                    "Unit_Hash": unit_hash,
                                    "processing_status": ProcessingStatus.SUCCESS,
                                    "Incident": analysis_result.get("Incident", ""),
                                    "Frame_SolutionRecommendation": analysis_result.get("Frame_SolutionRecommendation", []),
                                    "Frame_ResponsibilityAttribution": analysis_result.get("Frame_ResponsibilityAttribution", []),
                                    "Frame_CausalExplanation": analysis_result.get("Frame_CausalExplanation", []),
                                    "Frame_MoralEvaluation": analysis_result.get("Frame_MoralEvaluation", []),
                                    "Frame_ProblemDefinition": analysis_result.get("Frame_ProblemDefinition", []),
                                    "Frame_ActionStatement": analysis_result.get("Frame_ActionStatement", []),
                                    "Valence": analysis_result.get("Valence", ""),
                                    "Evidence_Type": analysis_result.get("Evidence_Type", ""),
                                    "Attribution_Level": analysis_result.get("Attribution_Level", ""),
                                    "Temporal_Focus": analysis_result.get("Temporal_Focus", ""),
                                    "Primary_Actor_Type": analysis_result.get("Primary_Actor_Type", ""),
                                    "Geographic_Scope": analysis_result.get("Geographic_Scope", ""),
                                    "Relationship_Model_Definition": analysis_result.get("Relationship_Model_Definition", ""),
                                    "Discourse_Type": analysis_result.get("Discourse_Type", ""),
                                }
                            )

                        if retry_records:
                            UX.ok(f"ID {original_id} | 续传完成：重新分析了{len(retry_records)}个单元")
                            return original_id, retry_records

                    else:
                        UX.warn(f"ID {original_id} | 未找到失败单元，将执行完整处理")
                except Exception as e:
                    UX.warn(f"ID {original_id} | 断点续传出错，将执行完整处理")

            # 完整的两阶段处理（首次处理或复用失败时）
            # 第一阶段：议题单元提取
            extraction_model = self.config.get('model_pools', {}).get('media_text_extraction', {}).get('primary')

            extraction_prompt = self.prompts.UNIT_EXTRACTION.replace("{full_text}", full_text)
            
            extraction_result = await self.api_service.get_analysis(
                extraction_prompt, 'analyzed_Units', language,
                stage_key='media_text_extraction',
                context_label=f"{original_id}:EXTRACTION"
            )
            
            if extraction_result is None:
                UX.warn(f"ID {original_id} | [阶段1/2] 提取失败")
                failed_record = create_unified_record(
                    ProcessingStatus.STAGE_1_FAILED,
                    original_id,
                    source,
                    unit_id=original_id
                )
                return original_id, [failed_record]
            
            if not isinstance(extraction_result, list) or not extraction_result:
                UX.info(f"ID {original_id} | [阶段1/2] 完成，未提取到议题单元")
                no_relevant_record = create_unified_record(
                    ProcessingStatus.NO_RELEVANT, original_id, source,
                    unit_text=full_text,  # 保存完整原文
                    extra_fields={"Article_ID": original_id, "text": full_text}
                )
                return original_id, [no_relevant_record]
            
            # 第二阶段：单元深度分析
            final_data = []
            
            for i, unit_data in enumerate(extraction_result, 1):
                unit_id = f"{original_id}-Unit-{i}"
                        
                unit_text = safe_str_convert(unit_data.get('Unit_Text', '')).strip()
                if not unit_text:
                    UX.warn(f"{unit_id} | 议题单元文本为空，跳过")
                    continue
                                
                seed_sentence = safe_str_convert(unit_data.get('seed_sentence', ''))
                expansion_logic = safe_str_convert(unit_data.get('expansion_logic', ''))
                # 清理旧的错误标记，确保不继承 API_CALL_FAILED
                raw_speaker = safe_str_convert(unit_data.get('speaker', ''))
                unit_speaker = raw_speaker if raw_speaker and raw_speaker != 'API_CALL_FAILED' else identify_source(source)
                
                analysis_prompt = self.prompts.UNIT_ANALYSIS.replace("{speaker}", unit_speaker).replace("{unit_text}", unit_text)
                
                analysis_result = await self.api_service.get_analysis(
                    analysis_prompt, None, language,
                    stage_key='media_text_analysis',
                    context_label=f"{unit_id}:ANALYSIS"
                )
                
                if analysis_result is None:
                    # 失败静默处理，最终汇总会体现
                    final_data.append(
                        create_unified_record(
                            ProcessingStatus.STAGE_2_FAILED,
                            original_id,
                            source,
                            unit_id=unit_id,
                            unit_text=unit_text,
                            speaker=unit_speaker,
                            extra_fields={
                                "seed_sentence": seed_sentence,
                                "expansion_logic": expansion_logic,
                                "Unit_Hash": "",
                            },
                        )
                    )
                    continue
                            
                # 合并第一阶段和第二阶段的结果
                norm_text = normalize_text(unit_text)
                unit_hash = hashlib.sha256(norm_text.encode('utf-8')).hexdigest()
                
                complete_unit = {
                    "Unit_ID": unit_id,
                    "Source": source,
                    "speaker": unit_speaker,
                    "Unit_Text": unit_text,
                    "seed_sentence": seed_sentence,
                    "expansion_logic": expansion_logic,
                    "Unit_Hash": unit_hash,
                    "processing_status": ProcessingStatus.SUCCESS, 
                    "Incident": analysis_result.get("Incident", ""),
                    "Frame_SolutionRecommendation": analysis_result.get("Frame_SolutionRecommendation", []),
                    "Frame_ResponsibilityAttribution": analysis_result.get("Frame_ResponsibilityAttribution", []),
                    "Frame_CausalExplanation": analysis_result.get("Frame_CausalExplanation", []),
                    "Frame_MoralEvaluation": analysis_result.get("Frame_MoralEvaluation", []),
                    "Frame_ProblemDefinition": analysis_result.get("Frame_ProblemDefinition", []),
                    "Frame_ActionStatement": analysis_result.get("Frame_ActionStatement", []),
                    "Valence": analysis_result.get("Valence", ""),
                    "Evidence_Type": analysis_result.get("Evidence_Type", ""),
                    "Attribution_Level": analysis_result.get("Attribution_Level", ""),
                    "Temporal_Focus": analysis_result.get("Temporal_Focus", ""),
                    "Primary_Actor_Type": analysis_result.get("Primary_Actor_Type", ""),
                    "Geographic_Scope": analysis_result.get("Geographic_Scope", ""),
                    "Relationship_Model_Definition": analysis_result.get("Relationship_Model_Definition", ""),
                    "Discourse_Type": analysis_result.get("Discourse_Type", "")
                }
                
                final_data.append(complete_unit)

            if not final_data:
                UX.info(f"ID {original_id} | [阶段2/2] 完成，无成功处理的单元")
                no_relevant_record = create_unified_record(
                    ProcessingStatus.NO_RELEVANT, original_id, source,
                    unit_text=full_text,  # 保存完整原文
                    extra_fields={"Article_ID": original_id, "text": full_text}
                )
                return original_id, [no_relevant_record]

            # 统计成功和失败单元数（只在有失败时提醒用户）
            success_count = len([u for u in final_data if u.get('processing_status') == ProcessingStatus.SUCCESS])
            failed_count = len([u for u in final_data if u.get('processing_status') == ProcessingStatus.STAGE_2_FAILED])
            if failed_count > 0:
                UX.warn(f"ID {original_id} | {failed_count} 个单元分析失败（共 {len(final_data)} 个单元）")
            return original_id, final_data

        except Exception as e:
            UX.err(f"处理ID {original_id} 时发生错误: {e}")
            failed_record = create_unified_record(
                ProcessingStatus.STAGE_2_FAILED, original_id, source, 
                "", f"错误: {str(e)[:100]}"
            )
            return original_id, [failed_record]
    
    
    def get_stage_model(self, stage_key):
        """获取阶段模型"""
        try:
            # 优先尝试从媒体文本专用模型池获取
            media_config = self.config.get('media_text', {})
            model_pools = media_config.get('model_pools', {})
            if model_pools and 'primary_models' in model_pools:
                return model_pools['primary_models'].get(stage_key)
            
            # 备用：尝试从API通用模型配置获取
            api_config = self.config.get('api', {})
            models = api_config.get('models', {})
            stage_mapping = {
                'UNIT_EXTRACTION': 'media_text_extraction',
                'UNIT_ANALYSIS': 'media_text_analysis'
            }
            model_key = stage_mapping.get(stage_key, stage_key.lower())
            if model_key in models:
                return models[model_key]
            
            # 向后兼容：尝试旧的配置结构
            MODEL_POOLS = self.config.get('model_pools', {})
            if MODEL_POOLS and 'primary_models' in MODEL_POOLS:
                return MODEL_POOLS['primary_models'][stage_key]
            # 使用统一的媒体文本配置
            media_config = self.config.get('media_text', {})
            model_pools = media_config.get('model_pools', {})
            return model_pools.get('primary_models', {})[stage_key]
        except Exception:
            raise ValueError(f"未配置阶段模型: {stage_key}")