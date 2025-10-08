import os
import hashlib

import pandas as pd

from .utils import UX, safe_str_convert, normalize_text, ProcessingStatus

# 从配置中获取阈值常量的辅助函数
def _get_thresholds_from_config(config):
    """从配置中提取阈值常量"""
    social_config = config.get('social_media', {})
    thresholds = social_config.get('text_length_thresholds', {})
    return {
        'ZHIHU_SHORT_TOKEN_THRESHOLD': thresholds.get('zhihu_short_text', 100),
        'ZHIHU_LONG_TOKEN_THRESHOLD': thresholds.get('zhihu_long_text', 1300),
        'VK_LONG_TEXT_THRESHOLD': thresholds.get('vk_long_text', 1500)
    }

class BaseProcessor:
    def __init__(self, api_service, config=None, prompts=None):
        self.api_service = api_service
        self.config = config or {}
        self.prompts = prompts
        self.Units_collector = []
        
        # 从配置中获取阈值常量
        self.thresholds = _get_thresholds_from_config(self.config)

    def _ensure_required_columns(self, df, extra_columns=None):
        """确保结果DataFrame包含统一输出列和额外列"""
        required_columns = list(self.config.get('required_output_columns', []) or [])
        if extra_columns is not None:
            for col in list(extra_columns):
                if col and col not in required_columns:
                    required_columns.append(col)

        if df is None or df.empty:
            if required_columns:
                return pd.DataFrame(columns=required_columns)
            return pd.DataFrame()

        for col in required_columns:
            if col not in df.columns:
                df[col] = pd.NA

        return df
    
    def _reorder_columns(self, df):
        """重新排列DataFrame列顺序，使输出更易读"""
        if df is None or df.empty:
            return df
        
        # 智能识别数据源类型
        is_zhihu = '知乎问题标题及描述' in df.columns or '回答内容' in df.columns
        is_vk = 'comment_id' in df.columns and 'post_id' in df.columns
        is_media = '序号' in df.columns and 'text' in df.columns and not is_zhihu
        
        # 定义不同数据源的列顺序优先级
        if is_zhihu:
            # 知乎列顺序
            column_order_priority = [
                # 1. 源数据列
                'Source', 'speaker', '知乎问题标题及描述', '回答内容', '回答用户名', '序号',
                # 2. 映射列（保留用于兼容）
                'Question', 'Answer_Text', 'Author',
                # 3. Unit内容列
                'Unit_ID', 'Unit_Hash', 'Unit_Text', 'processing_status',
            ]
        elif is_vk:
            # VK列顺序
            column_order_priority = [
                # 1. 源数据列
                'Source', 'channel_name', 'comment_id', 'post_id', 'comment_text',
                # 2. Unit内容列
                'Unit_ID', 'Unit_Hash', 'post_text', 'Unit_Text', 'processing_status',
            ]
        else:
            # 媒体文本列顺序
            column_order_priority = [
                # 1. 源数据列（输入文件原有列）
                'Source', '序号', '日期', '标题', '出版社', 'Token数', 'text',
                # 2. Source信息列
                'speaker', 'seed_sentence', 'expansion_logic',
                # 3. Unit内容列
                'Unit_ID', 'Unit_Hash', 'Unit_Text', 'processing_status',
            ]
        
        # 通用分析字段（所有类型共用）
        analysis_fields = [
            'Incident',
            'Frame_SolutionRecommendation', 'Frame_ResponsibilityAttribution',
            'Frame_CausalExplanation', 'Frame_MoralEvaluation',
            'Frame_ProblemDefinition', 'Frame_ActionStatement',
            'Valence', 'Evidence_Type', 'Attribution_Level',
            'Temporal_Focus', 'Primary_Actor_Type', 'Geographic_Scope',
            'Relationship_Model_Definition', 'Discourse_Type',
            'relevance',
        ]
        column_order_priority.extend(analysis_fields)
        
        # 构建最终列顺序：优先列 + 其他列（按原顺序）
        ordered_cols = [col for col in column_order_priority if col in df.columns]
        remaining_cols = [col for col in df.columns if col not in ordered_cols]
        final_order = ordered_cols + remaining_cols
        
        return df.reindex(columns=final_order)

    def _upsert_dataframe(self, df_new, output_path, key_columns=None):
        """将新数据合并至输出文件，按键列去重保留最新"""
        if df_new is None or df_new.empty:
            return pd.DataFrame()

        df_new = df_new.copy()
        df_new = self._ensure_required_columns(df_new, extra_columns=df_new.columns)

        try:
            if os.path.exists(output_path):
                df_existing = pd.read_excel(output_path)
            else:
                df_existing = pd.DataFrame()
        except Exception as e:
            UX.warn(f"读取结果文件失败({output_path}): {e}")
            df_existing = pd.DataFrame()

        df_existing = self._ensure_required_columns(df_existing, extra_columns=df_new.columns)

        combined = pd.concat([df_existing, df_new], ignore_index=True)

        key_columns = [col for col in (key_columns or []) if col in combined.columns]
        if key_columns:
            for key in key_columns:
                combined[key] = combined[key].astype(str)
            combined.drop_duplicates(subset=key_columns, keep='last', inplace=True)

        combined = self._ensure_required_columns(combined, extra_columns=df_new.columns)
        combined.reset_index(drop=True, inplace=True)
        combined = self._reorder_columns(combined)
        combined.to_excel(output_path, index=False)
        return combined

    def _save_records_to_output(self, records, output_path, key_columns=None):
        """保存字典记录列表至输出文件"""
        if not records:
            return pd.DataFrame()
        df_records = pd.DataFrame(records)
        return self._upsert_dataframe(df_records, output_path, key_columns=key_columns)

    def _compute_unit_hash(self, text):
        """基于标准化文本计算单元哈希"""
        normalized_text = normalize_text(safe_str_convert(text))
        return hashlib.sha256(normalized_text.encode('utf-8')).hexdigest()
    
    def _add_hash_to_record(self, record, text_field):
        """为记录添加哈希值（基于指定文本字段）"""
        if not isinstance(record, dict):
            return record

        text = safe_str_convert(record.get(text_field, ''))
        record['Unit_Hash'] = self._compute_unit_hash(text)
        return record

    def load_failed_units(self, output_path):
        """加载需要重新分析的单元（STAGE_2_FAILED）
        
        返回：[{Unit_ID, Unit_Text, ...}, ...]
        """
        df = self._read_output_dataframe(output_path)
        if df.empty or 'processing_status' not in df.columns:
            return []
        
        mask = df['processing_status'] == ProcessingStatus.STAGE_2_FAILED
        return df[mask].to_dict(orient='records')

    def load_failed_stage1_ids(self, output_path, id_column):
        """加载第一阶段失败的ID（STAGE_1_FAILED）
        
        返回：{'id1', 'id2', ...}
        """
        df = self._read_output_dataframe(output_path)
        if df.empty or 'processing_status' not in df.columns or id_column not in df.columns:
            return set()
        
        mask = df['processing_status'] == ProcessingStatus.STAGE_1_FAILED
        return set(df[mask][id_column].astype(str))

    def get_never_processed_ids(self, output_path, input_df, id_column):
        """获取从未处理过的ID
        
        返回：{'id1', 'id2', ...}
        """
        all_input_ids = set(input_df[id_column].astype(str))
        
        df = self._read_output_dataframe(output_path)
        if df.empty or id_column not in df.columns:
            return all_input_ids
        
        processed_ids = set(df[id_column].astype(str))
        return all_input_ids - processed_ids

    def _read_output_dataframe(self, output_path):
        """读取输出文件，失败时返回空DataFrame"""
        if not output_path or not os.path.exists(output_path):
            return pd.DataFrame()

        try:
            df_existing = pd.read_excel(output_path)
        except Exception as e:
            UX.warn(f"读取结果文件失败({output_path}): {e}")
            return pd.DataFrame()

        if not isinstance(df_existing, pd.DataFrame):
            return pd.DataFrame()

        return df_existing

    def _write_stage_one_records(self, output_path, stage_one_rows, mapping):
        """写入第一阶段生成的PENDING记录，按评论ID/Unit_ID去重后落盘"""
        if not stage_one_rows:
            return self._read_output_dataframe(output_path)

        df_new = pd.DataFrame(stage_one_rows)
        df_new = self._ensure_required_columns(df_new, extra_columns=df_new.columns)

        df_existing = self._read_output_dataframe(output_path)
        if not df_existing.empty:
            df_existing = self._ensure_required_columns(df_existing, extra_columns=df_new.columns)

            # 针对知乎阶段一失败的记录：若本次写入同一回答的阶段一结果，移除旧的 STAGE_1_FAILED 行，避免阻塞断点续传
            if 'processing_status' in df_existing.columns:
                identifiers = set()
                for row in stage_one_rows:
                    identifier = None
                    if mapping:
                        id_col = mapping.get('id')
                        if id_col:
                            identifier = row.get(id_col)
                    if identifier is not None:
                        identifiers.add(safe_str_convert(identifier))

                if identifiers:
                    candidate_cols = []
                    if mapping:
                        id_col = mapping.get('id')  # 直接使用序号列
                        if id_col and id_col in df_existing.columns:
                            candidate_cols.append(id_col)

                    drop_mask = (df_existing['processing_status'] == ProcessingStatus.STAGE_1_FAILED)

                    if candidate_cols:
                        match_mask = pd.Series(False, index=df_existing.index)
                        for col in candidate_cols:
                            match_mask |= df_existing[col].astype(str).isin(identifiers)
                        drop_mask &= match_mask

                    if 'Unit_ID' in df_existing.columns:
                        # 阶段一失败时，Unit_ID 等于 序号，无后缀
                        drop_mask |= df_existing['Unit_ID'].astype(str).isin(identifiers)

                    if drop_mask.any():
                        df_existing = df_existing[~drop_mask]

        combined = pd.concat([df_existing, df_new], ignore_index=True)

        dedup_columns = []
        comment_id_col = mapping.get('comment_id') if mapping else None
        if comment_id_col and comment_id_col in combined.columns:
            dedup_columns.append(comment_id_col)
        if 'Unit_ID' in combined.columns:
            dedup_columns.append('Unit_ID')

        dedup_columns = [col for col in dedup_columns if col in combined.columns]
        for col in dedup_columns:
            combined[col] = combined[col].astype(str)

        if dedup_columns:
            combined.drop_duplicates(subset=dedup_columns, keep='last', inplace=True)

        combined = self._ensure_required_columns(combined, extra_columns=df_new.columns)
        combined.reset_index(drop=True, inplace=True)
        combined = self._reorder_columns(combined)
        combined.to_excel(output_path, index=False)

        return combined

    def _save_stage_two_results(self, df_units, output_path, mapping):
        """保存第二阶段分析结果，按Unit_ID覆盖旧记录"""
        if df_units is None or df_units.empty:
            return self._read_output_dataframe(output_path)

        df_units = df_units.copy()
        df_units = self._ensure_required_columns(df_units, extra_columns=df_units.columns)
        if 'relevance' not in df_units.columns:
            df_units['relevance'] = pd.NA

        # 确保增强后的记录不会覆盖状态（尤其是当外部手动改写后）
        status_col = 'processing_status'
        if status_col not in df_units.columns:
            df_units[status_col] = ProcessingStatus.SUCCESS

        df_existing = self._read_output_dataframe(output_path)
        if not df_existing.empty:
            df_existing = self._ensure_required_columns(df_existing, extra_columns=df_units.columns)

        combined = pd.concat([df_existing, df_units], ignore_index=True)

        dedup_columns = ['Unit_ID']
        comment_id_col = mapping.get('comment_id') if mapping else None
        if comment_id_col and comment_id_col in combined.columns:
            dedup_columns.append(comment_id_col)

        dedup_columns = [col for col in dedup_columns if col in combined.columns]
        for col in dedup_columns:
            combined[col] = combined[col].astype(str)

        if dedup_columns:
            combined.drop_duplicates(subset=dedup_columns, keep='last', inplace=True)

        combined = self._ensure_required_columns(combined, extra_columns=df_units.columns)
        combined.reset_index(drop=True, inplace=True)
        combined = self._reorder_columns(combined)
        combined.to_excel(output_path, index=False)

        return combined