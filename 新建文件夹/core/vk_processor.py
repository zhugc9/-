import asyncio
import hashlib
import json
import pandas as pd
from tqdm.asyncio import tqdm as aio_tqdm
from .base_processor import BaseProcessor
from .utils import (
    UX,
    safe_str_convert,
    normalize_text,
    count_tokens,
    create_unified_record,
    ProcessingStatus,
    ANALYSIS_COLUMNS,
)

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
                # 如果存在阶段二失败状态，则需要重新加入
                elif ProcessingStatus.STAGE_2_FAILED in status_values:
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
                    'processing_status': ProcessingStatus.STAGE_2_FAILED,
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
                    'processing_status': ProcessingStatus.STAGE_2_FAILED,
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


    async def process(self, df, output_path, source='vk'):
        """处理VK数据集"""
        UX.phase("VK评论处理")

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

        # 🔍 断点续传计划
        UX.resume_plan("VK数据")
        
        total_comments = len(df)
        existing_success = 0
        existing_no_relevant = 0
        if not existing_df.empty and 'processing_status' in existing_df.columns:
            existing_success = (existing_df['processing_status'] == ProcessingStatus.SUCCESS).sum()
            existing_no_relevant = (existing_df['processing_status'] == ProcessingStatus.NO_RELEVANT).sum()
        existing_completed = existing_success + existing_no_relevant
        
        # VK处理逻辑：要么处理新评论，要么处理失败单元（不会同时处理）
        new_comments = len(stage_one_records) if stage_one_records else 0
        retry_comments = len(self.load_failed_units(output_path)) if not stage_one_records else 0
        
        UX.info(f"总评论数: {total_comments} | 已完成: {existing_completed}（成功 {existing_success} + 无关 {existing_no_relevant}）")
        
        if new_comments > 0:
            UX.info(f"本次处理: 新评论 {new_comments} 条")
        elif retry_comments > 0:
            UX.info(f"本次处理: 失败续传 {retry_comments} 条")
        
        UX.resume_end()

        if stage_one_records:
            UX.info(f"  [阶段1/2] 准备 {len(stage_one_records)} 条评论记录")
            self._write_stage_one_records(output_path, stage_one_records, mapping)
        else:
            UX.info("  [阶段1/2] 无新评论需要准备")

        pending_units = pending_units or self.load_failed_units(output_path)

        if not pending_units:
            UX.ok("VK处理完成，无待分析评论")
            return

        df_units = pd.DataFrame(pending_units)
        UX.info(f"  [阶段2/2] 开始批量分析 {len(df_units)} 条评论")

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
            UX.warn("无可处理的批次任务")
            return

        UX.info(f"  批处理任务数: {len(batch_tasks)}")

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
                UX.warn(f"批次分析失败，涉及 {len(expected_comment_ids)} 条评论，这批次中的{len(expected_comment_ids)}条评论全部失败")
                for comment_id in expected_comment_ids:
                    mask = df_units[mapping['comment_id']] == comment_id
                    df_units.loc[mask, 'processing_status'] = ProcessingStatus.STAGE_2_FAILED
                    df_units.loc[mask, 'relevance'] = 'API调用失败'
                continue

            if not isinstance(batch_results, list):
                UX.warn(f"批次分析格式错误，涉及 {len(expected_comment_ids)} 条评论，这批次中的{len(expected_comment_ids)}条评论全部失败")
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
                UX.warn(f"{len(missing_ids)} 条评论未返回分析结果，这{len(missing_ids)}条评论已标记为失败")
                for comment_id in missing_ids:
                    mask = df_units[mapping['comment_id']] == comment_id
                    df_units.loc[mask, 'processing_status'] = ProcessingStatus.STAGE_2_FAILED
                    if 'relevance' in df_units.columns:
                        df_units.loc[mask, 'relevance'] = '未返回结果'

            save_interval = vk_config.get('save_interval', 5)
            if save_interval and (batch_idx + 1) % save_interval == 0:
                self._save_stage_two_results(df_units, output_path, mapping)
                success_count = (df_units['processing_status'] == ProcessingStatus.SUCCESS).sum()
                failed_count = (df_units['processing_status'] == ProcessingStatus.STAGE_2_FAILED).sum()
                no_relevant_count = (df_units['processing_status'] == ProcessingStatus.NO_RELEVANT).sum()
                total_processed = success_count + failed_count + no_relevant_count
                progress_rate = (total_processed / max(len(df_units), 1)) * 100
                UX.info(f"  进度保存 {progress_rate:.1f}% | 成功 {success_count} | 失败 {failed_count} | 无相关 {no_relevant_count}")

        progress.close()

        self._save_stage_two_results(df_units, output_path, mapping)

        success_count = (df_units['processing_status'] == ProcessingStatus.SUCCESS).sum()
        failed_count = (df_units['processing_status'] == ProcessingStatus.STAGE_2_FAILED).sum()
        no_relevant_count = (df_units['processing_status'] == ProcessingStatus.NO_RELEVANT).sum()

        total = len(df_units)
        completion_rate = ((success_count + no_relevant_count) / max(total, 1)) * 100
        UX.ok(f"VK处理完成 | 完成度 {completion_rate:.1f}%")
        UX.info(f"  统计: 成功 {success_count} | 无相关 {no_relevant_count} | 失败 {failed_count} | 总计 {total}")

        if failed_count > 0:
            UX.warn(f"  仍有 {failed_count} 条评论处理失败，可再次运行重试")

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
            else:
                stage_key = 'vk_batch_standard'

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

                if not status:
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
        return create_unified_record(
            ProcessingStatus.STAGE_2_FAILED,
            comment_id,
            'vk',
            unit_id=f"VK-{comment_id}",
            unit_text="",
            speaker="",
            extra_fields={
                'comment_id': comment_id,
                'relevance': 'API处理失败',
                'Incident': reason,
            },
        )
    
    def _add_hash_to_record(self, record, text_field):
        """为记录添加哈希值"""
        text = safe_str_convert(record.get(text_field, ''))
        normalized_text = normalize_text(text)
        record['Unit_Hash'] = hashlib.sha256(normalized_text.encode('utf-8')).hexdigest()
        return record
