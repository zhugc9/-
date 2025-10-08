import json

import pandas as pd

from .base_processor import BaseProcessor
from .utils import (
    UX,
    safe_str_convert,
    count_tokens,
    create_unified_record,
    ProcessingStatus,
)


class ZhihuProcessor(BaseProcessor):
    """知乎回答处理器 - 两阶段断点续传"""
    
    def _get_author_name(self, original_row, mapping):
        author_column = mapping.get('author', '回答用户名')
        if author_column in original_row.index:
            author_value = safe_str_convert(original_row[author_column])
            return author_value.strip() or '未知作者'
        return '未知作者'
    
    def _finalize_record(self, result_data, original_row, mapping, answer_id, unit_index=1):
        if not isinstance(result_data, dict):
            return None

        author = self._get_author_name(original_row, mapping)
        
        result_data['speaker'] = author
        result_data['Source'] = '知乎'
        result_data.setdefault('processing_status', ProcessingStatus.SUCCESS)
        result_data['Unit_ID'] = f"{answer_id}-Unit-{unit_index}"  # 统一格式：序号-Unit-编号
        result_data['序号'] = answer_id  # 保持与输入列名一致

        self._add_hash_to_record(result_data, 'Unit_Text')
        return result_data
    
    def _record_unit_meta(self, unit_id, source, question, answer_text, author, unit_text=None, status=None):
        for entry in self.Units_collector:
            if entry.get('Unit_ID') == unit_id:
                if status is not None:
                    entry['AI_Is_Relevant'] = (status == ProcessingStatus.SUCCESS)
                if unit_text and 'Unit_Text' not in entry:
                    entry['Unit_Text'] = unit_text[:500]
                return

        meta = {
            'Unit_ID': unit_id,
            'Source': source,
            'Question': question,
            'Author': author or '未知作者',
            'AI_Is_Relevant': None if status is None else status == ProcessingStatus.SUCCESS
        }

        if unit_text is not None:
            meta['Unit_Text'] = unit_text[:500]
        else:
            meta['Answer_Text'] = answer_text[:500]

        self.Units_collector.append(meta)
    
    def _extract_unit_index(self, unit_id):
        try:
            if not unit_id:
                return 1

            unit_id_str = safe_str_convert(unit_id)
            parts = unit_id_str.split('-')
            if len(parts) >= 3 and parts[-1].isdigit():
                return int(parts[-1])
            # 对旧格式或异常格式返回1，保持向后兼容
            return 1
        except Exception:
            return 1

    async def process(self, df, output_path, source='知乎'):
        UX.phase("知乎回答处理")
        
        social_config = self.config.get('social_media', {})
        mapping = (social_config.get('column_mapping') or {}).get('zhihu', {})

        answer_id_col = mapping.get('id')
        question_col = mapping.get('question')
        answer_text_col = mapping.get('answer_text')
        author_col = mapping.get('author', '回答用户名')

        required_cols = {answer_id_col, question_col, answer_text_col}
        if None in required_cols or '' in required_cols:
            UX.err("知乎列映射缺少必要字段(id/question/answer_text)")
            return []

        missing = required_cols - set(df.columns)
        if missing:
            UX.err(f"知乎文件缺少必要列: {list(missing)}")
            return []

        df = df.copy()
        df[answer_id_col] = df[answer_id_col].astype(str)

        existing_df = self._read_output_dataframe(output_path)

        short_records = []
        long_records_stage_one = []

        for _, row in df.iterrows():
            answer_id = safe_str_convert(row[answer_id_col])
            question = safe_str_convert(row[question_col])
            answer_text = safe_str_convert(row[answer_text_col])
            author = self._get_author_name(row, mapping)

            if not answer_text.strip():
                record = create_unified_record(ProcessingStatus.NO_RELEVANT, answer_id, source)
                record.update({
                    'Unit_ID': f"ZH-{answer_id}-1",
                    answer_id_col: answer_id,
                    question_col: question,
                    answer_text_col: answer_text,
                    'Question': question,
                    'Answer_Text': answer_text,
                    author_col: author,
                    'Author': author,
                    'Unit_Text': '[无相关内容]'
                })
                short_records.append(record)
                self._record_unit_meta(f"ZH-{answer_id}-1", source, question, answer_text, author, unit_text='[无相关内容]', status=ProcessingStatus.NO_RELEVANT)
                continue

            token_count = count_tokens(answer_text)
            if token_count < self.thresholds['ZHIHU_SHORT_TOKEN_THRESHOLD']:
                short_outputs = await self._process_short_answer(row, mapping, answer_id, question, answer_text, author, source)
                if short_outputs:
                    short_records.extend(short_outputs)
                continue

            self._record_unit_meta(f"ZH-{answer_id}", source, question, answer_text, author)
            long_records_stage_one.append({
                'row': row,
                'answer_id': answer_id,
                'question': question,
                'answer_text': answer_text,
                'author': author
            })

        stage_one_rows = await self._prepare_zhihu_stage_one(long_records_stage_one, existing_df, mapping, source)
        
        # 🔍 断点续传计划
        UX.resume_plan("知乎数据")
        
        total_answers = len(df)
        existing_completed_units = 0
        if not existing_df.empty and 'processing_status' in existing_df.columns:
            existing_completed_units = (
                (existing_df['processing_status'] == ProcessingStatus.SUCCESS) | 
                (existing_df['processing_status'] == ProcessingStatus.NO_RELEVANT)
            ).sum()
        
        # 计算实际要处理的数量
        short_count = len(short_records)
        answer_id_col = mapping.get('id', '序号')
        long_count = len(set([row.get(answer_id_col) or row.get('序号') for row in stage_one_rows if row.get(answer_id_col) or row.get('序号')])) if stage_one_rows else 0
        long_units_count = len(stage_one_rows)
        
        total_to_process = short_count + long_count
        
        UX.info(f"总回答数: {total_answers} | 已完成单元: {existing_completed_units}")
        if long_units_count > 0 and long_count > 0:
            UX.info(f"本次处理: {total_to_process} 篇回答（{long_units_count} 个单元待分析）")
        else:
            UX.info(f"本次处理: {total_to_process} 篇回答")
        
        UX.resume_end()
        
        # 先写入阶段1结果
        if stage_one_rows:
            UX.info(f"  [阶段1/2] 切分完成，生成 {len(stage_one_rows)} 个议题单元")
            self._write_stage_one_records(output_path, stage_one_rows, mapping)
        
        # 再加载待处理单元（包括刚写入的阶段1单元）
        pending_units = self.load_failed_units(output_path)

        if pending_units:
            UX.info(f"  [阶段2/2] 开始分析 {len(pending_units)} 个待处理单元")
            df_pending = pd.DataFrame(pending_units)
            df_pending = self._ensure_required_columns(df_pending, extra_columns=df_pending.columns)
            df_pending = await self._run_stage_two(df_pending, mapping, source)
            self._save_stage_two_results(df_pending, output_path, mapping)
        else:
            UX.info("  [阶段2/2] 无待处理单元")

        if short_records:
            df_short = pd.DataFrame(short_records)
            df_short = self._ensure_required_columns(df_short, extra_columns=df_short.columns)
            self._save_stage_two_results(df_short, output_path, mapping)

        df_final = self._read_output_dataframe(output_path)
        if not df_final.empty and 'processing_status' in df_final.columns:
            success_cnt = (df_final['processing_status'] == ProcessingStatus.SUCCESS).sum()
            norelv_cnt = (df_final['processing_status'] == ProcessingStatus.NO_RELEVANT).sum()
            failed_cnt = (df_final['processing_status'] == ProcessingStatus.STAGE_2_FAILED).sum()
            total = len(df_final)
            completion_rate = ((success_cnt + norelv_cnt) / max(total, 1)) * 100
            
            UX.ok(f"知乎处理完成 | 完成度 {completion_rate:.1f}%")
            UX.info(f"  统计: 成功 {success_cnt} | 无相关 {norelv_cnt} | 失败 {failed_cnt} | 总计 {total}")
            if failed_cnt:
                UX.warn(f"  仍有 {failed_cnt} 个单元处理失败，可再次运行重试")
        else:
            UX.ok("知乎处理完成")

        return []

    async def _process_short_answer(self, row, mapping, answer_id, question, answer_text, author, source):
        prompt = self.prompts.ZHIHU_SHORT_ANALYSIS.format(question=question, answer_text=answer_text)
        result = await self.api_service.call_api_async(prompt, 'zh', 'ZHIHU_ANALYSIS_SHORT')

        answer_id_col = mapping.get('id')
        question_col = mapping.get('question')
        answer_text_col = mapping.get('answer_text')
        author_col = mapping.get('author', '回答用户名')

        unit_id = f"ZH-{answer_id}-1"

        if not result or not isinstance(result, dict):
            record = create_unified_record(ProcessingStatus.STAGE_2_FAILED, answer_id, source, answer_text[:200], '知乎短文本分析失败')
            record.update({
                'Unit_ID': unit_id,
                answer_id_col: answer_id,
                question_col: question,
                answer_text_col: answer_text,
                'Question': question,
                'Answer_Text': answer_text,
                author_col: author,
                'Author': author,
                'Unit_Text': answer_text,
                'speaker': author
            })
            self._record_unit_meta(unit_id, source, question, answer_text, author, unit_text=answer_text, status=ProcessingStatus.STAGE_2_FAILED)
            return [record]

        status = result.get('processing_status')
        if not status:
            status = ProcessingStatus.NO_RELEVANT if result.get('relevance') == '不相关' else ProcessingStatus.SUCCESS
        result['processing_status'] = status
        result.setdefault('relevance', None)
        result['Unit_Text'] = answer_text
        result['Question'] = question
        result['Answer_Text'] = answer_text
        result[question_col] = question
        result[answer_text_col] = answer_text
        result[answer_id_col] = answer_id
        result['Author'] = author
        result[author_col] = author

        finalized = self._finalize_record(result, row, mapping, answer_id, unit_index=1)
        if not finalized:
                return []
            
        for key, value in list(finalized.items()):
            if isinstance(value, list):
                finalized[key] = json.dumps(value, ensure_ascii=False)

        finalized.update({
            question_col: question,
            answer_text_col: answer_text,
            author_col: author,
            'Question': question,
            'Answer_Text': answer_text,
            'Author': author
        })

        self._record_unit_meta(unit_id, source, question, answer_text, author, unit_text=answer_text, status=status)
        return [finalized]

    async def _prepare_zhihu_stage_one(self, stage_one_inputs, existing_df, mapping, source):
        if not stage_one_inputs:
            return []

        answer_id_col = mapping.get('id')
        question_col = mapping.get('question')
        answer_text_col = mapping.get('answer_text')
        author_col = mapping.get('author', '回答用户名')

        existing_df = existing_df if isinstance(existing_df, pd.DataFrame) else pd.DataFrame()
        if not existing_df.empty and answer_id_col in existing_df.columns:
            existing_df[answer_id_col] = existing_df[answer_id_col].astype(str)

        existing_groups = {}
        if not existing_df.empty and answer_id_col in existing_df.columns:
            for ans_id, group in existing_df.groupby(answer_id_col):
                existing_groups[str(ans_id)] = group

        stage_one_rows = []

        for item in stage_one_inputs:
            row = item['row']
            answer_id = item['answer_id']
            question = item['question']
            answer_text = item['answer_text']
            author = item['author']

            existing_group = existing_groups.get(answer_id)
            reuse_rows = pd.DataFrame()
            if existing_group is not None:
                # 检查是否有阶段一失败：需要重新切分，不能复用
                has_stage1_failed = (existing_group['processing_status'] == ProcessingStatus.STAGE_1_FAILED).any()
                
                if not has_stage1_failed:
                    # 没有阶段一失败，检查是否有阶段二失败的单元
                    retry_mask = existing_group['processing_status'] == ProcessingStatus.STAGE_2_FAILED
                    reuse_rows = existing_group[retry_mask]
                    if reuse_rows.empty:
                        # 已全部完成
                        self._record_unit_meta(f"ZH-{answer_id}", source, question, answer_text, author, status=ProcessingStatus.SUCCESS)
                        continue
                # 如果有阶段一失败，reuse_rows保持为空DataFrame，继续执行后续的重新切分逻辑

            if not reuse_rows.empty and 'Unit_Text' in reuse_rows.columns and reuse_rows['Unit_Text'].notna().all():
                reuse_rows = reuse_rows.sort_values('Unit_ID')
                for _, unit_row in reuse_rows.iterrows():
                    unit_id = safe_str_convert(unit_row.get('Unit_ID'))
                    unit_text = safe_str_convert(unit_row.get('Unit_Text'))
                    if not unit_id or not unit_text.strip():
                        continue
                    stage_one_rows.append({
                        'Unit_ID': unit_id,
                        answer_id_col: answer_id,
                        question_col: question,
                        answer_text_col: answer_text,
                        author_col: author,
                        'Question': question,
                        'Answer_Text': answer_text,
                        'Author': author,
                        'Unit_Text': unit_text,
                        'processing_status': ProcessingStatus.STAGE_2_FAILED,
                        'Source': source
                    })
                    self._record_unit_meta(unit_id, source, question, answer_text, author, unit_text=unit_text)
                continue

            chapters = await self._chunk_answer(answer_text)
            if not chapters:
                stage_one_rows.append({
                    'Unit_ID': answer_id,  # 阶段1失败时Unit_ID=序号
                    answer_id_col: answer_id,
                    question_col: question,
                    answer_text_col: answer_text,
                    author_col: author,
                    'Question': question,
                    'Answer_Text': answer_text,
                    'Author': author,
                    'Unit_Text': '',
                    'processing_status': ProcessingStatus.STAGE_1_FAILED,
                    'Source': source,
                    'speaker': author
                })
                self._record_unit_meta(answer_id, source, question, answer_text, author, status=ProcessingStatus.STAGE_1_FAILED)
                continue

            for idx, chapter in enumerate(chapters, start=1):
                unit_text = safe_str_convert(chapter.get('Unit_Text'))
                if not unit_text.strip():
                    continue
                unit_id = f"{answer_id}-Unit-{idx}"  # 统一格式
                stage_one_rows.append({
                    'Unit_ID': unit_id,
                    answer_id_col: answer_id,
                    question_col: question,
                    answer_text_col: answer_text,
                    author_col: author,
                    'Question': question,
                    'Answer_Text': answer_text,
                    'Author': author,
                    'Unit_Text': unit_text,
                    'processing_status': ProcessingStatus.STAGE_2_FAILED,
                    'Source': source
                })
                self._record_unit_meta(unit_id, source, question, answer_text, author, unit_text=unit_text)

        return stage_one_rows

    async def _chunk_answer(self, answer_text):
        prompt = self.prompts.ZHIHU_CHUNKING.format(full_text=answer_text)
        result = await self.api_service.call_api_async(prompt, 'zh', 'ZHIHU_CHUNKING')
        if not result:
            return []
        if isinstance(result, dict):
            return result.get('argument_chapters', []) or []
        if isinstance(result, list):
            return result
        return []


    async def _run_stage_two(self, df_units, mapping, source):
        answer_id_col = mapping.get('id')
        question_col = mapping.get('question')
        answer_text_col = mapping.get('answer_text')
        author_col = mapping.get('author', '回答用户名')

        # 预先将需要写入文本/JSON的列转换为object类型，避免pandas未来版本的类型限制
        text_like_columns = [
            'Unit_ID', 'Unit_Text', 'Question', 'Answer_Text', 'Author', 'speaker', 'Unit_Hash',
            'relevance', 'Incident', 'Valence', 'Evidence_Type', 'Attribution_Level',
            'Temporal_Focus', 'Primary_Actor_Type', 'Geographic_Scope',
            'Relationship_Model_Definition', 'Discourse_Type'
        ] + [
            'Frame_SolutionRecommendation', 'Frame_ResponsibilityAttribution',
            'Frame_CausalExplanation', 'Frame_MoralEvaluation',
            'Frame_ProblemDefinition', 'Frame_ActionStatement'
        ]

        for col in text_like_columns:
            if col in df_units.columns:
                df_units[col] = df_units[col].astype('object')
            else:
                df_units[col] = pd.Series([pd.NA] * len(df_units), dtype='object')

        answer_stats = {}

        for idx, unit_row in df_units.iterrows():
            unit_id = safe_str_convert(unit_row.get('Unit_ID'))
            answer_id = safe_str_convert(unit_row.get(answer_id_col) or (unit_id.split('-')[1] if unit_id and '-' in unit_id else ''))
            question = safe_str_convert(unit_row.get('Question') or unit_row.get(question_col))
            answer_text = safe_str_convert(unit_row.get('Answer_Text') or unit_row.get(answer_text_col))
            author = safe_str_convert(unit_row.get('Author') or unit_row.get(author_col)) or '未知作者'
            unit_text = safe_str_convert(unit_row.get('Unit_Text'))

            if not unit_text.strip():
                df_units.at[idx, 'processing_status'] = ProcessingStatus.NO_RELEVANT
                df_units.at[idx, 'relevance'] = '空章节'
                self._record_unit_meta(unit_id, source, question, answer_text, author, unit_text=unit_text, status=ProcessingStatus.NO_RELEVANT)
                continue

            prompt = self.prompts.ZHIHU_ANALYSIS.format(question=question, Unit_Text=unit_text)

            unit_tokens = count_tokens(unit_text)
            if unit_tokens < self.thresholds['ZHIHU_SHORT_TOKEN_THRESHOLD']:
                stage_key = 'ZHIHU_ANALYSIS_SHORT'
            elif unit_tokens > self.thresholds['ZHIHU_LONG_TOKEN_THRESHOLD']:
                stage_key = 'ZHIHU_ANALYSIS_LONG'
            else:
                stage_key = 'ZHIHU_ANALYSIS'

            result = await self.api_service.call_api_async(prompt, 'zh', stage_key)

            if not result or not isinstance(result, dict):
                df_units.at[idx, 'processing_status'] = ProcessingStatus.STAGE_2_FAILED
                df_units.at[idx, 'Incident'] = 'API调用失败' if not result else '响应格式无效'
                self._record_unit_meta(unit_id, source, question, answer_text, author, unit_text=unit_text, status=ProcessingStatus.STAGE_2_FAILED)
                continue

            status = result.get('processing_status')
            if not status:
                status = ProcessingStatus.NO_RELEVANT if result.get('relevance') == '不相关' else ProcessingStatus.SUCCESS

            result.update({
                'processing_status': status,
                'Unit_Text': unit_text,
                'Question': question,
                'Answer_Text': answer_text,
                question_col: question,
                answer_text_col: answer_text,
                answer_id_col: answer_id,
                'Author': author,
                author_col: author
            })

            if 'relevance' not in result:
                result['relevance'] = None

            finalized = self._finalize_record(result, unit_row, mapping, answer_id, self._extract_unit_index(unit_id))
            if not finalized:
                df_units.at[idx, 'processing_status'] = ProcessingStatus.STAGE_2_FAILED
                df_units.at[idx, 'Incident'] = '结果解析失败'
                self._record_unit_meta(unit_id, source, question, answer_text, author, unit_text=unit_text, status=ProcessingStatus.STAGE_2_FAILED)
                continue

            for key, value in list(finalized.items()):
                if isinstance(value, list):
                    finalized[key] = json.dumps(value, ensure_ascii=False)
                df_units.at[idx, key] = finalized[key]

            df_units.at[idx, 'Question'] = question
            df_units.at[idx, 'Answer_Text'] = answer_text
            df_units.at[idx, question_col] = question
            df_units.at[idx, answer_text_col] = answer_text
            df_units.at[idx, author_col] = author
            df_units.at[idx, 'Author'] = author

            self._record_unit_meta(unit_id, source, question, answer_text, author, unit_text=unit_text, status=finalized.get('processing_status'))

            stats = answer_stats.setdefault(answer_id, {'success': 0, 'valid': 0})
            if finalized.get('processing_status') == ProcessingStatus.SUCCESS:
                stats['success'] += 1
            if finalized.get('processing_status') in {ProcessingStatus.SUCCESS, ProcessingStatus.NO_RELEVANT}:
                stats['valid'] += 1

        # 更新回答级别的相关性
        for meta in self.Units_collector:
            unit_id = meta.get('Unit_ID')
            if unit_id and unit_id.startswith('ZH-') and unit_id.count('-') == 1:
                answer_id = unit_id.split('-')[1]
                stats = answer_stats.get(answer_id)
                if stats:
                    meta['AI_Is_Relevant'] = stats['success'] > 0

        return df_units