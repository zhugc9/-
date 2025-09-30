# -*- coding: utf-8 -*-
"""
Processor classes for handling different data sources and analysis strategies.
"""
import os
import asyncio
import hashlib
import pandas as pd
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from tqdm.asyncio import tqdm as aio_tqdm
from .utils import (
    UX, safe_str_convert, normalize_text, count_tokens, identify_source,
    get_processing_state, create_unified_record, detect_language_and_get_config,
    ProcessingStatus, ANALYSIS_COLUMNS
)
from .api_service import APIService

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
                    identifier = row.get('Answer_ID')
                    if not identifier and mapping:
                        id_col = mapping.get('id')
                        if id_col:
                            identifier = row.get(id_col)
                    if identifier is not None:
                        identifiers.add(safe_str_convert(identifier))

                if identifiers:
                    candidate_cols = []
                    if 'Answer_ID' in df_existing.columns:
                        candidate_cols.append('Answer_ID')
                    if mapping:
                        id_col = mapping.get('id')
                        if id_col and id_col in df_existing.columns:
                            candidate_cols.append(id_col)

                    drop_mask = (df_existing['processing_status'] == ProcessingStatus.STAGE_1_FAILED)

                    if candidate_cols:
                        match_mask = pd.Series(False, index=df_existing.index)
                        for col in candidate_cols:
                            match_mask |= df_existing[col].astype(str).isin(identifiers)
                        drop_mask &= match_mask

                    if 'Unit_ID' in df_existing.columns:
                        stage1_failed_unit_ids = {f"{answer_id}-STAGE1_FAILED" for answer_id in identifiers}
                        drop_mask |= df_existing['Unit_ID'].astype(str).isin(stage1_failed_unit_ids)

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
        combined.to_excel(output_path, index=False)

        UX.info(f"阶段一结果写入完成，当前总行数: {len(combined)}")
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
        combined.to_excel(output_path, index=False)

        UX.info(f"阶段二结果写入完成，当前总行数: {len(combined)}")
        return combined

class VKProcessor(BaseProcessor):
    """VK评论处理器 - 异步版本"""

    def _prepare_vk_stage_one(self, df_input, existing_df, mapping, source):
        """准备VK第一阶段记录并返回待分析单元"""

        stage_one_rows = []
        pending_units = []

        if df_input is None or df_input.empty:
            return stage_one_rows, pending_units

        comment_id_col = mapping.get('comment_id')
        comment_text_col = mapping.get('comment_text')
        post_id_col = mapping.get('post_id')
        post_text_col = mapping.get('post_text')
        channel_name_col = mapping.get('channel_name')

        existing_df = existing_df if isinstance(existing_df, pd.DataFrame) else pd.DataFrame()
        existing_df = self._ensure_required_columns(existing_df, extra_columns=['Unit_ID', 'processing_status'])

        existing_status_map = {}
        if not existing_df.empty and comment_id_col in existing_df.columns:
            for _, row in existing_df.iterrows():
                comment_id = safe_str_convert(row.get(comment_id_col))
                if not comment_id:
                    continue
                status = row.get('processing_status')
                unit_id = safe_str_convert(row.get('Unit_ID'))
                existing_status_map.setdefault(comment_id, []).append((status, unit_id))

        for _, row in df_input.iterrows():
            comment_id = safe_str_convert(row.get(comment_id_col))
            comment_text = safe_str_convert(row.get(comment_text_col))
            post_id = safe_str_convert(row.get(post_id_col))
            post_text = safe_str_convert(row.get(post_text_col))
            channel_name = safe_str_convert(row.get(channel_name_col)) if channel_name_col else ''

            unit_id = f"VK-{comment_id}"

            existing_statuses = existing_status_map.get(comment_id, [])
            should_add = True
            if existing_statuses:
                status_values = [status for status, _ in existing_statuses if status]
                # 如果已经有SUCCESS/NO_RELEVANT记录，则跳过重新写入
                if any(status in [ProcessingStatus.SUCCESS, ProcessingStatus.NO_RELEVANT] for status in status_values):
                    should_add = False
                # 如果存在阶段二失败或待分析状态，则需要重新加入
                elif any(status in [ProcessingStatus.STAGE_2_FAILED, ProcessingStatus.PENDING_ANALYSIS] for status in status_values):
                    should_add = True
                else:
                    should_add = True

            if should_add and comment_text.strip():
                # 记录阶段一待分析结果
                stage_one_row = {
                    comment_id_col: comment_id,
                    'Unit_ID': unit_id,
                    'Source': source,
                    post_id_col: post_id,
                    post_text_col: post_text,
                    comment_text_col: comment_text,
                    'Unit_Text': comment_text,
                    'processing_status': ProcessingStatus.PENDING_ANALYSIS,
                }
                if channel_name_col:
                    stage_one_row[channel_name_col] = channel_name
                stage_one_rows.append(stage_one_row)

                # 自动加入第二阶段待处理队列
                pending_unit = {
                    comment_id_col: comment_id,
                    comment_text_col: comment_text,
                    post_id_col: post_id,
                    post_text_col: post_text,
                    'Unit_ID': unit_id,
                    'Source': source,
                    'Unit_Text': comment_text,
                }
                if channel_name_col:
                    pending_unit[channel_name_col] = channel_name
                pending_units.append(pending_unit)
            elif should_add and not comment_text.strip():
                # 空评论直接标记为无关
                stage_one_row = {
                    comment_id_col: comment_id,
                    'Unit_ID': unit_id,
                    'Source': source,
                    post_id_col: post_id,
                    post_text_col: post_text,
                    comment_text_col: comment_text,
                    'Unit_Text': comment_text,
                    'processing_status': ProcessingStatus.NO_RELEVANT,
                    'relevance': '空评论'
                }
                if channel_name_col:
                    stage_one_row[channel_name_col] = channel_name
                stage_one_rows.append(stage_one_row)

        return stage_one_rows, pending_units

    def _load_vk_pending_units(self, output_path, mapping):
        """从结果表中加载待处理的VK单元"""

        df_existing = self._read_output_dataframe(output_path)
        if df_existing.empty:
            return []

        df_existing = self._ensure_required_columns(df_existing, extra_columns=['Unit_ID', 'processing_status'])

        required_status = {ProcessingStatus.PENDING_ANALYSIS, ProcessingStatus.STAGE_2_FAILED}
        comment_id_col = mapping.get('comment_id')
        comment_text_col = mapping.get('comment_text')
        post_id_col = mapping.get('post_id')
        post_text_col = mapping.get('post_text')
        channel_name_col = mapping.get('channel_name')

        pending_mask = df_existing['processing_status'].isin(required_status)
        df_pending = df_existing[pending_mask].copy()

        pending_units = []
        for _, row in df_pending.iterrows():
            comment_text = safe_str_convert(row.get(comment_text_col))
            pending_unit = {
                comment_id_col: safe_str_convert(row.get(comment_id_col)),
                comment_text_col: comment_text,
                post_id_col: safe_str_convert(row.get(post_id_col)),
                post_text_col: safe_str_convert(row.get(post_text_col)),
                'Unit_ID': safe_str_convert(row.get('Unit_ID')),
                'processing_status': safe_str_convert(row.get('processing_status')),
                'Source': safe_str_convert(row.get('Source')),
                'Unit_Text': comment_text,
                'relevance': safe_str_convert(row.get('relevance')) if 'relevance' in df_pending.columns else pd.NA,
            }
            if channel_name_col and channel_name_col in df_pending.columns:
                pending_unit[channel_name_col] = safe_str_convert(row.get(channel_name_col))
            pending_units.append(pending_unit)

        return pending_units

    async def process(self, df, output_path, source='vk'):
        """处理VK数据集"""
        UX.info("处理VK评论...")

        social_config = self.config.get('social_media', {})
        column_mapping = social_config.get('column_mapping', {})
        mapping = column_mapping.get('vk', {})

        required_columns = set(mapping.values())
        actual_columns = set(df.columns)
        if not required_columns.issubset(actual_columns):
            missing = required_columns - actual_columns
            UX.err(f"VK文件格式错误，缺少以下必需列: {list(missing)}。已跳过此文件。")
            return

        df[mapping['comment_id']] = df[mapping['comment_id']].astype(str)

        existing_df = self._read_output_dataframe(output_path)
        stage_one_records, pending_units = self._prepare_vk_stage_one(df, existing_df, mapping, source)

        if stage_one_records:
            UX.info(f"阶段一写入准备 {len(stage_one_records)} 条记录")
            self._write_stage_one_records(output_path, stage_one_records, mapping)
        else:
            UX.info("阶段一未发现需要写入的新记录")

        pending_units = pending_units or self._load_vk_pending_units(output_path, mapping)

        if not pending_units:
            UX.ok("VK第一阶段完成，无单元需要分析")
            return

        df_units = pd.DataFrame(pending_units)
        UX.info(f"待分析单元: {len(df_units)} 条")

        vk_config = social_config.get('vk_processing', {})

        analysis_columns = ANALYSIS_COLUMNS + ['Unit_Hash', 'relevance']
        df_units['Source'] = source
        for col in analysis_columns:
            if col not in df_units.columns:
                df_units[col] = pd.NA

        api_strategy = self.config.get('api', {}).get('strategy', {})
        max_concurrent = api_strategy.get('max_concurrent_requests', 2)

        batch_tasks = []
        batch_comment_map = {}

        for post_id, group in df_units.groupby(mapping['post_id'], dropna=False):
            if pd.isna(post_id):
                continue

            post_text = safe_str_convert(group[mapping['post_text']].iloc[0])
            comments_payload = []
            comment_ids_in_batch = []

            for _, row in group.iterrows():
                comment_id = str(row[mapping['comment_id']])
                comment_text = safe_str_convert(row[mapping['comment_text']])

                if not comment_text.strip():
                    mask = df_units[mapping['comment_id']] == comment_id
                    df_units.loc[mask, 'processing_status'] = ProcessingStatus.NO_RELEVANT
                    df_units.loc[mask, 'relevance'] = '空评论'
                    continue

                comments_payload.append({
                    "comment_id": comment_id,
                    "comment_text": comment_text
                })
                comment_ids_in_batch.append(comment_id)

                channel_name_col = mapping.get('channel_name')
                channel_name = safe_str_convert(row.get(channel_name_col, '未知频道')) if channel_name_col in group.columns else '未知频道'

                self.Units_collector.append({
                    'Unit_ID': f"VK-{comment_id}",
                    'Source': source,
                    'Post_ID': str(post_id),
                    'Post_Text': post_text,
                    'Comment_Text': comment_text,
                    'channel_name': channel_name,
                    'AI_Is_Relevant': None
                })

            batch_size_limit = vk_config.get('batch_size_limit', 20)
            for i in range(0, len(comments_payload), batch_size_limit):
                chunk = comments_payload[i:i + batch_size_limit]
                if not chunk:
                    continue
                batch_idx = len(batch_tasks)
                batch_tasks.append({
                    'post_id': str(post_id),
                    'post_text': post_text,
                    'comments': chunk
                })
                batch_comment_map[batch_idx] = comment_ids_in_batch[i:i + batch_size_limit]

        if not batch_tasks:
            UX.warn("没有可处理的VK批次任务")
            return

        UX.info(f"创建了 {len(batch_tasks)} 个批处理任务")

        semaphore = asyncio.Semaphore(max_concurrent)
        progress = aio_tqdm(total=len(batch_tasks), desc="VK批处理", position=0)

        async def run_batch(idx, batch_task):
            async with semaphore:
                results = await self._process_batch(batch_task)
                return idx, results

        tasks = [asyncio.create_task(run_batch(idx, task)) for idx, task in enumerate(batch_tasks)]

        for task in asyncio.as_completed(tasks):
            batch_idx, batch_results = await task
            progress.update(1)
            expected_comment_ids = batch_comment_map.get(batch_idx, [])

            if batch_results is None:
                UX.warn(f"批次 {batch_idx} 返回None，标记所有评论为失败")
                for comment_id in expected_comment_ids:
                    mask = df_units[mapping['comment_id']] == comment_id
                    df_units.loc[mask, 'processing_status'] = ProcessingStatus.STAGE_2_FAILED
                    df_units.loc[mask, 'relevance'] = 'API调用失败'
                continue

            if not isinstance(batch_results, list):
                UX.warn(f"批次 {batch_idx} 返回非列表结果")
                for comment_id in expected_comment_ids:
                    mask = df_units[mapping['comment_id']] == comment_id
                    df_units.loc[mask, 'processing_status'] = ProcessingStatus.STAGE_2_FAILED
                    df_units.loc[mask, 'relevance'] = '返回格式错误'
                continue

            processed_comment_ids = set()

            for result in batch_results:
                if not isinstance(result, dict):
                    continue

                comment_id = str(result.get('comment_id', ''))
                if not comment_id:
                    continue

                processed_comment_ids.add(comment_id)
                mask = df_units[mapping['comment_id']] == comment_id
                if not mask.any():
                    continue

                original_text = safe_str_convert(df_units.loc[mask, mapping['comment_text']].iloc[0])
                result.setdefault('comment_text', original_text)

                status = result.get('processing_status')
                relevance = result.get('relevance')
                if not status:
                    if relevance == '不相关':
                        status = ProcessingStatus.NO_RELEVANT
                    elif relevance in ['API_FAILED', 'INVALID_RESPONSE', 'EXCEPTION']:
                        status = ProcessingStatus.STAGE_2_FAILED
                    else:
                        status = ProcessingStatus.SUCCESS

                df_units.loc[mask, 'processing_status'] = status
                if 'relevance' not in df_units.columns:
                    df_units['relevance'] = pd.NA
                df_units.loc[mask, 'relevance'] = relevance if (relevance is not None and relevance != '') else pd.NA
                if 'relevance' not in result or result.get('relevance') in (None, ''):
                    result['relevance'] = relevance if (relevance is not None and relevance != '') else None

                for unit in self.Units_collector:
                    if unit['Unit_ID'] == f"VK-{comment_id}":
                        unit['AI_Is_Relevant'] = status == ProcessingStatus.SUCCESS
                        break

                for key, value in result.items():
                    if key in (mapping['comment_id'], 'processing_status', 'relevance'):
                        continue
                    target_key = key
                    if isinstance(value, list):
                        value = json.dumps(value, ensure_ascii=False)
                    if target_key not in df_units.columns:
                        df_units[target_key] = pd.NA
                    df_units.loc[mask, target_key] = value

            missing_ids = set(expected_comment_ids) - processed_comment_ids
            if missing_ids:
                UX.warn(f"批次 {batch_idx} 中有 {len(missing_ids)} 个评论未返回结果，标记为失败")
                for comment_id in missing_ids:
                    mask = df_units[mapping['comment_id']] == comment_id
                    df_units.loc[mask, 'processing_status'] = ProcessingStatus.STAGE_2_FAILED
                    if 'relevance' in df_units.columns:
                        df_units.loc[mask, 'relevance'] = '未返回结果'

            save_interval = vk_config.get('save_interval', 5)
            if save_interval and (batch_idx + 1) % save_interval == 0:
                self._save_stage_two_results(df_units, output_path, mapping)
                success_count = (df_units['processing_status'] == ProcessingStatus.SUCCESS).sum()
                failed_count = (df_units['processing_status'].isin({ProcessingStatus.STAGE_2_FAILED})).sum()
                no_relevant_count = (df_units['processing_status'] == ProcessingStatus.NO_RELEVANT).sum()
                pending_count = (df_units['processing_status'] == ProcessingStatus.PENDING_ANALYSIS).sum()
                total_processed = success_count + failed_count + no_relevant_count
                progress_rate = (total_processed / max(len(df_units), 1)) * 100
                UX.info(f"📊 VK处理进度 ({progress_rate:.1f}%): 成功{success_count}, 失败{failed_count}, 无相关{no_relevant_count}, 待处理{pending_count}")

        progress.close()

        pending_mask = df_units['processing_status'] == ProcessingStatus.PENDING_ANALYSIS
        if pending_mask.any():
            UX.warn(f"发现 {pending_mask.sum()} 条未处理记录，标记为失败")
            df_units.loc[pending_mask, 'processing_status'] = ProcessingStatus.STAGE_2_FAILED
            df_units.loc[pending_mask, 'relevance'] = '未被处理'

        self._save_stage_two_results(df_units, output_path, mapping)

        success_count = (df_units['processing_status'] == ProcessingStatus.SUCCESS).sum()
        failed_count = (df_units['processing_status'] == ProcessingStatus.STAGE_2_FAILED).sum()
        no_relevant_count = (df_units['processing_status'] == ProcessingStatus.NO_RELEVANT).sum()

        completion_rate = ((success_count + no_relevant_count) / max(len(df_units), 1)) * 100
        UX.ok(f"📋 VK处理完成总结: 完成度{completion_rate:.1f}% (成功{success_count} + 无相关{no_relevant_count})")
        UX.info(f"   📊 详细统计: 成功{success_count}, 失败{failed_count}, 无相关{no_relevant_count}")

        if failed_count > 0:
            UX.warn(f"   ⚠️  仍有{failed_count}条记录处理失败，可再次运行进行智能重试")

    async def _process_batch(self, batch_task):
        try:
            post_text = batch_task['post_text']
            comments_json = json.dumps(batch_task['comments'], ensure_ascii=False)

            prompt = self.prompts.VK_BATCH_ANALYSIS.format(
                post_text=str(post_text),
                comments_json=comments_json
            )

            combined_text = f"{post_text} {comments_json}"
            tokens = count_tokens(combined_text)
            if tokens > self.thresholds['VK_LONG_TEXT_THRESHOLD']:
                stage_key = 'vk_batch_long'
                UX.info(f"🔧 VK批次模型选择: ({tokens} tokens) → 长文本模型")
            else:
                stage_key = 'vk_batch_standard'
                UX.info(f"🔧 VK批次模型选择: ({tokens} tokens) → 标准模型")

            result = await self.api_service.call_api_async(prompt, 'ru', stage_key)

            if result is None:
                UX.warn(f"API调用返回None，批次包含 {len(batch_task['comments'])} 条评论")
                post_id = batch_task.get('post_id')
                return [self._create_failed_record(c['comment_id'], 'API返回None', post_id)
                        for c in batch_task['comments']]

            processed_results = self._normalize_batch_result(result, batch_task)
            if not processed_results:
                UX.warn(f"无法从API响应中提取有效结果，批次包含 {len(batch_task['comments'])} 条评论")
                post_id = batch_task.get('post_id')
                return [self._create_failed_record(c['comment_id'], 'API响应格式无效', post_id)
                        for c in batch_task['comments']]

            expected_ids = {str(c['comment_id']) for c in batch_task['comments']}
            result_ids = {str(item.get('comment_id')) for item in processed_results if item.get('comment_id')}

            missing_ids = expected_ids - result_ids
            if missing_ids:
                UX.warn(f"API响应缺少 {len(missing_ids)} 条评论的结果")
                post_id = batch_task.get('post_id')
                for comment_id in missing_ids:
                    processed_results.append(self._create_failed_record(comment_id, 'API响应中缺失', post_id))

            extra_ids = result_ids - expected_ids
            if extra_ids:
                UX.warn(f"API响应包含 {len(extra_ids)} 条额外评论，将被忽略")
                processed_results = [item for item in processed_results if str(item.get('comment_id')) in expected_ids]

            return processed_results

        except Exception as e:
            UX.err(f"批次处理异常: {str(e)}")
            post_id = batch_task.get('post_id')
            return [self._create_failed_record(c['comment_id'], f'异常: {str(e)[:50]}', post_id)
                    for c in batch_task['comments']]

    def _normalize_batch_result(self, result, batch_task):
        processed_results = []

        if isinstance(result, dict):
            for key in ['analysis', 'results', 'processed_results', 'data', 'comments']:
                if key in result and isinstance(result[key], list):
                    result = result[key]
                    break

        if isinstance(result, list):
            for item in result:
                if not isinstance(item, dict):
                    continue

                comment_id = str(item.get('comment_id', ''))
                if not comment_id:
                    continue

                for comment in batch_task['comments']:
                    if str(comment['comment_id']) == comment_id:
                        item.setdefault('comment_text', comment['comment_text'])
                        item.setdefault('Unit_Text', comment['comment_text'])
                        break

                status = item.get('processing_status')
                relevance = item.get('relevance')

                if not status or status == ProcessingStatus.PENDING_ANALYSIS:
                    if relevance == '不相关':
                        item['processing_status'] = ProcessingStatus.NO_RELEVANT
                    elif relevance in ['API_FAILED', 'INVALID_RESPONSE', 'EXCEPTION']:
                        item['processing_status'] = ProcessingStatus.STAGE_2_FAILED
                    else:
                        item['processing_status'] = ProcessingStatus.SUCCESS

                if 'relevance' not in item:
                    item['relevance'] = relevance if relevance is not None else None
                elif item['relevance'] is None:
                    item['relevance'] = relevance

                self._add_hash_to_record(item, 'comment_text')
                processed_results.append(item)

        return processed_results

    def _create_failed_record(self, comment_id, reason, post_id=None):
        """创建失败记录"""
        record = create_unified_record(ProcessingStatus.STAGE_2_FAILED, comment_id, 'vk', '', reason)
        record['comment_id'] = comment_id
        record['relevance'] = 'API_FAILED'
        record['speaker'] = 'API_CALL_FAILED'
        record['Incident'] = reason
        
        if post_id:
            record['Batch_ID'] = f"{post_id}-BATCH_FAILED"
        
        return record
    
    def _add_hash_to_record(self, record, text_field):
        """为记录添加哈希值"""
        text = safe_str_convert(record.get(text_field, ''))
        normalized_text = normalize_text(text)
        record['Unit_Hash'] = hashlib.sha256(normalized_text.encode('utf-8')).hexdigest()
        return record


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
        result_data['Unit_ID'] = f"ZH-{answer_id}-{unit_index}"
        result_data['Answer_ID'] = f"ZH-{answer_id}"
        result_data['id'] = answer_id
        result_data['序号'] = answer_id

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
        UX.info("处理知乎回答...")
        
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
                    'Answer_ID': f"ZH-{answer_id}",
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
        if stage_one_rows:
            UX.info(f"知乎阶段一写入 {len(stage_one_rows)} 条记录")
            self._write_stage_one_records(output_path, stage_one_rows, mapping)

        pending_units = self._load_zhihu_pending_units(output_path)

        if pending_units:
            df_pending = pd.DataFrame(pending_units)
            df_pending = self._ensure_required_columns(df_pending, extra_columns=df_pending.columns)
            df_pending = await self._run_stage_two(df_pending, mapping, source)
            self._save_stage_two_results(df_pending, output_path, mapping)
        else:
            UX.info("知乎阶段二无待处理单元")

        if short_records:
            df_short = pd.DataFrame(short_records)
            df_short = self._ensure_required_columns(df_short, extra_columns=df_short.columns)
            self._save_stage_two_results(df_short, output_path, mapping)

        df_final = self._read_output_dataframe(output_path)
        if not df_final.empty and 'processing_status' in df_final.columns:
            success_cnt = (df_final['processing_status'] == ProcessingStatus.SUCCESS).sum()
            norelv_cnt = (df_final['processing_status'] == ProcessingStatus.NO_RELEVANT).sum()
            failed_cnt = (df_final['processing_status'] == ProcessingStatus.STAGE_2_FAILED).sum()
            pending_cnt = (df_final['processing_status'] == ProcessingStatus.PENDING_ANALYSIS).sum()
            UX.ok(f"知乎处理完成: 成功{success_cnt}条, 无相关{norelv_cnt}条, 二阶段失败{failed_cnt}条, 待处理{pending_cnt}条")
            if failed_cnt or pending_cnt:
                UX.warn("知乎仍有章节待重试，可再次运行")
        else:
            UX.ok("知乎处理完成，无可写入结果")

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
                'Answer_ID': f"ZH-{answer_id}",
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
        result['Answer_ID'] = f"ZH-{answer_id}"
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
                retry_mask = existing_group['processing_status'].isin({ProcessingStatus.PENDING_ANALYSIS, ProcessingStatus.STAGE_2_FAILED})
                reuse_rows = existing_group[retry_mask]
                if reuse_rows.empty:
                    # 已全部完成
                    self._record_unit_meta(f"ZH-{answer_id}", source, question, answer_text, author, status=ProcessingStatus.SUCCESS)
                    continue

            if not reuse_rows.empty and 'Unit_Text' in reuse_rows.columns and reuse_rows['Unit_Text'].notna().all():
                reuse_rows = reuse_rows.sort_values('Unit_ID')
                for _, unit_row in reuse_rows.iterrows():
                    unit_id = safe_str_convert(unit_row.get('Unit_ID'))
                    unit_text = safe_str_convert(unit_row.get('Unit_Text'))
                    if not unit_id or not unit_text.strip():
                        continue
                    stage_one_rows.append({
                        'Unit_ID': unit_id,
                        'Answer_ID': f"ZH-{answer_id}",
                        answer_id_col: answer_id,
                        question_col: question,
                        answer_text_col: answer_text,
                        author_col: author,
                        'Question': question,
                        'Answer_Text': answer_text,
                        'Author': author,
                        'Unit_Text': unit_text,
                        'processing_status': ProcessingStatus.PENDING_ANALYSIS,
                        'Source': source
                    })
                    self._record_unit_meta(unit_id, source, question, answer_text, author, unit_text=unit_text)
                continue

            chapters = await self._chunk_answer(answer_text)
            if not chapters:
                stage_one_rows.append({
                    'Unit_ID': f"ZH-{answer_id}-STAGE1_FAILED",
                    'Answer_ID': f"ZH-{answer_id}",
                    answer_id_col: answer_id,
                    question_col: question,
                    answer_text_col: answer_text,
                    author_col: author,
                    'Question': question,
                    'Answer_Text': answer_text,
                    'Author': author,
                    'Unit_Text': '[切分失败]',
                    'processing_status': ProcessingStatus.STAGE_1_FAILED,
                    'Incident': '知乎切分失败',
                    'Source': source,
                    'speaker': author
                })
                self._record_unit_meta(f"ZH-{answer_id}", source, question, answer_text, author, status=ProcessingStatus.STAGE_1_FAILED)
                continue

            for idx, chapter in enumerate(chapters, start=1):
                unit_text = safe_str_convert(chapter.get('Unit_Text'))
                if not unit_text.strip():
                    continue
                unit_id = f"ZH-{answer_id}-{idx}"
                stage_one_rows.append({
                    'Unit_ID': unit_id,
                    'Answer_ID': f"ZH-{answer_id}",
                    answer_id_col: answer_id,
                    question_col: question,
                    answer_text_col: answer_text,
                    author_col: author,
                    'Question': question,
                    'Answer_Text': answer_text,
                    'Author': author,
                    'Unit_Text': unit_text,
                    'processing_status': ProcessingStatus.PENDING_ANALYSIS,
                    'Source': source,
                    'expansion_logic': chapter.get('expansion_logic')
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

    def _load_zhihu_pending_units(self, output_path):
        df_existing = self._read_output_dataframe(output_path)
        if df_existing.empty or 'processing_status' not in df_existing.columns:
            return []

        mask = df_existing['processing_status'].isin({ProcessingStatus.PENDING_ANALYSIS, ProcessingStatus.STAGE_2_FAILED})
        df_pending = df_existing[mask].copy()
        if df_pending.empty:
            return []
        df_pending = df_pending.sort_values('Unit_ID') if 'Unit_ID' in df_pending.columns else df_pending
        return df_pending.to_dict(orient='records')

    async def _run_stage_two(self, df_units, mapping, source):
        answer_id_col = mapping.get('id')
        question_col = mapping.get('question')
        answer_text_col = mapping.get('answer_text')
        author_col = mapping.get('author', '回答用户名')

        # 预先将需要写入文本/JSON的列转换为object类型，避免pandas未来版本的类型限制
        text_like_columns = [
            'Unit_Text', 'Question', 'Answer_Text', 'Author', 'speaker', 'Unit_Hash',
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
                'Answer_ID': f"ZH-{answer_id}",
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
                no_relevant_record = create_unified_record(ProcessingStatus.NO_RELEVANT, original_id, source)
                return original_id, [no_relevant_record]

            # 检查文本长度限制
            language, config = detect_language_and_get_config(full_text, self.config)
            text_tokens = count_tokens(full_text)
            if text_tokens > config['MAX_SINGLE_TEXT']:
                UX.warn(f"ID {original_id}: 文本过长({text_tokens} tokens)，跳过处理")
                failed_record = create_unified_record(
                    ProcessingStatus.STAGE_1_FAILED, original_id, source, 
                    full_text[:200], f"文本过长({text_tokens} tokens)"
                )
                return original_id, [failed_record]
            
            UX.info(f"🚀 ID {original_id}: 开始两阶段处理 ({text_tokens} tokens, {language})")

            # 第一阶段：议题单元提取
            extraction_model = self.config.get('model_pools', {}).get('media_text_extraction', {}).get('primary')
            UX.info(f"📋 ID {original_id}: [1/2] 议题单元提取阶段 - 使用模型: {extraction_model or '未配置'}")

            extraction_prompt = self.prompts.UNIT_EXTRACTION.replace("{full_text}", full_text)
            
            extraction_result = await self.api_service.get_analysis(
                extraction_prompt, 'analyzed_Units', language,
                stage_key='media_text_extraction',
                context_label=f"{original_id}:EXTRACTION"
            )
            
            if extraction_result is None:
                UX.warn(f"❌ ID {original_id}: [1/2] 第一阶段议题单元提取失败")
                failed_record = create_unified_record(
                    ProcessingStatus.STAGE_1_FAILED, original_id, source, 
                    full_text[:200], "第一阶段议题单元提取失败"
                )
                return original_id, [failed_record]
            
            if not isinstance(extraction_result, list) or not extraction_result:
                UX.info(f"📝 ID {original_id}: [1/2] 完成，但未提取到任何议题单元")
                no_relevant_record = create_unified_record(ProcessingStatus.NO_RELEVANT, original_id, source)
                return original_id, [no_relevant_record]
            
            UX.ok(f"✅ ID {original_id}: [1/2] 完成，提取到 {len(extraction_result)} 个议题单元")

            # 第二阶段：单元深度分析
            UX.info(f"🔍 ID {original_id}: [2/2] 单元深度分析阶段 - 处理 {len(extraction_result)} 个单元")
            final_data = []
            
            for i, unit_data in enumerate(extraction_result, 1):
                unit_id = f"{original_id}-Unit-{i}"
                
                # 检查是否需要跳过已成功处理的单元（断点续传）
                if failed_units and unit_id not in failed_units:
                    UX.info(f"⏭️  {unit_id}: 跳过已成功处理的单元")
                    continue
                        
                unit_text = safe_str_convert(unit_data.get('Unit_Text', '')).strip()
                if not unit_text:
                    UX.warn(f"⚠️  {unit_id}: 议题单元文本为空，跳过")
                    continue
                                
                seed_sentence = safe_str_convert(unit_data.get('seed_sentence', ''))
                expansion_logic = safe_str_convert(unit_data.get('expansion_logic', ''))
                unit_speaker = safe_str_convert(unit_data.get('speaker', identify_source(source)))
                
                UX.info(f"📝 {unit_id}: 开始分析 (发言人: {unit_speaker[:20]}{'...' if len(unit_speaker) > 20 else ''})")
                
                analysis_prompt = self.prompts.UNIT_ANALYSIS.replace("{speaker}", unit_speaker).replace("{unit_text}", unit_text)
                
                analysis_result = await self.api_service.get_analysis(
                    analysis_prompt, None, language,
                    stage_key='media_text_analysis',
                    context_label=f"{unit_id}:ANALYSIS"
                )
                
                if analysis_result is None:
                    UX.warn(f"❌ {unit_id}: 第二阶段分析失败")
                    failed_unit = {
                        "processing_status": ProcessingStatus.STAGE_2_FAILED,
                        "Source": source,
                        "Unit_ID": unit_id,
                        "speaker": "API_CALL_FAILED",
                        "Unit_Text": f"[API_FAILED] 单元 {unit_id} 第二阶段分析失败: {unit_text[:200]}...",
                        "seed_sentence": seed_sentence,
                        "expansion_logic": expansion_logic,
                        "Unit_Hash": "",
                        "Incident": "API_CALL_FAILED",
                        "Frame_SolutionRecommendation": "[]",
                        "Frame_ResponsibilityAttribution": "[]",
                        "Frame_CausalExplanation": "[]",
                        "Frame_MoralEvaluation": "[]",
                        "Frame_ProblemDefinition": "[]",
                        "Frame_ActionStatement": "[]",
                        "Valence": "API_CALL_FAILED",
                        "Evidence_Type": "API_CALL_FAILED",
                        "Attribution_Level": "API_CALL_FAILED",
                        "Temporal_Focus": "API_CALL_FAILED",
                        "Primary_Actor_Type": "API_CALL_FAILED",
                        "Geographic_Scope": "API_CALL_FAILED",
                        "Relationship_Model_Definition": "API_CALL_FAILED",
                        "Discourse_Type": "API_CALL_FAILED"
                    }
                    final_data.append(failed_unit)
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
                UX.ok(f"✅ {unit_id}: 分析完成")

            if not final_data:
                UX.info(f"📝 ID {original_id}: [2/2] 完成，但没有成功处理的议题单元")
                no_relevant_record = create_unified_record(ProcessingStatus.NO_RELEVANT, original_id, source)
                return original_id, [no_relevant_record]

            UX.ok(f"🎉 ID {original_id}: [2/2] 完成！共生成 {len(final_data)} 个议题单元")
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