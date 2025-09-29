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
from tqdm.asyncio import tqdm as aio_tqdm, gather as aio_gather
from .utils import (
    UX, safe_str_convert, normalize_text, count_tokens, identify_source,
    get_processing_state, clean_failed_records, create_unified_record, detect_language_and_get_config,
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
    
    def _save_progress_generic(self, df_to_process, output_path, id_column):
        """通用保存进度方法"""
        if df_to_process.empty:
            return
        
        # 只保存成功和无相关的记录
        if 'processing_status' in df_to_process.columns:
            save_mask = df_to_process['processing_status'].isin([
                ProcessingStatus.SUCCESS, 
                ProcessingStatus.NO_RELEVANT
            ])
            df_to_save = df_to_process[save_mask].copy()
        else:
            # 兼容旧版本：排除API_CALL_FAILED
            if 'speaker' in df_to_process.columns:
                save_mask = ~df_to_process['speaker'].astype(str).str.contains('API_CALL_FAILED', na=False)
                df_to_save = df_to_process[save_mask].copy()
            else:
                df_to_save = df_to_process.copy()
        
        if df_to_save.empty:
            return
        
        # 精确合并或创建文件
        if os.path.exists(output_path):
            try:
                df_existing = pd.read_excel(output_path)
                if not df_existing.empty and id_column in df_existing.columns:
                    # 获取已成功处理的ID（不包括失败的）
                    success_existing_ids, _ = get_processing_state(df_existing, id_column)
                    
                    # 只添加新的成功记录，不覆盖已成功的
                    new_mask = ~df_to_save[id_column].astype(str).isin(success_existing_ids)
                    df_new = df_to_save[new_mask]
                    
                    if not df_new.empty:
                        df_final = pd.concat([df_existing, df_new], ignore_index=True)
                        UX.info(f"添加了 {len(df_new)} 条新记录到现有 {len(df_existing)} 条")
                    else:
                        df_final = df_existing
                        UX.info(f"无新记录需要添加，保持现有 {len(df_existing)} 条")
                else:
                    df_final = df_to_save
            except Exception as e:
                UX.warn(f"读取现有文件失败: {e}，使用新数据")
                df_final = df_to_save
        else:
            df_final = df_to_save
        
        df_final.to_excel(output_path, index=False)
        
        # 显示统计
        if 'processing_status' in df_final.columns:
            success_count = (df_final['processing_status'] == ProcessingStatus.SUCCESS).sum()
            no_relevant_count = (df_final['processing_status'] == ProcessingStatus.NO_RELEVANT).sum()
            failed_count = (df_final['processing_status'] == ProcessingStatus.API_FAILED).sum()
            UX.info(f"进度保存: 成功{success_count}条, 无相关{no_relevant_count}条, 失败{failed_count}条")

class VKProcessor(BaseProcessor):
    """VK评论处理器 - 修复版"""
    
    def process(self, df, output_path, source='vk'):
        """处理VK文件 - 修复版"""
        UX.info("处理VK评论...")
        
        # 获取VK列映射配置（统一路径）
        social_config = self.config.get('social_media', {})
        column_mapping = social_config.get('column_mapping', {})
        mapping = column_mapping.get('vk', {})
        
        # 输入数据模式校验
        required_columns = set(mapping.values())
        actual_columns = set(df.columns)
        
        if not required_columns.issubset(actual_columns):
            missing = required_columns - actual_columns
            UX.err(f"VK文件格式错误，缺少以下必需列: {list(missing)}。已跳过此文件。")
            return
        
        # 检查已处理的记录（包括成功和失败）
        processed_ids = set()
        failed_ids = set()
        if os.path.exists(output_path):
            try:
                df_existing_check = pd.read_excel(output_path)
                if not df_existing_check.empty and mapping['comment_id'] in df_existing_check.columns:
                    processed_ids, failed_ids = get_processing_state(df_existing_check, mapping['comment_id'])
                    
                    if 'processing_status' in df_existing_check.columns:
                        success_count = (df_existing_check['processing_status'] == ProcessingStatus.SUCCESS).sum()
                        no_relevant_count = (df_existing_check['processing_status'] == ProcessingStatus.NO_RELEVANT).sum()
                        failed_count = (df_existing_check['processing_status'] == ProcessingStatus.API_FAILED).sum()
                        UX.info(f"VK已处理: {len(processed_ids)} (成功: {success_count}, 无相关: {no_relevant_count})")
                        if failed_ids:
                            UX.info(f"发现VK失败记录: {len(failed_ids)} 个，将重新分析")
                    else:
                        UX.info(f"VK已处理: {len(processed_ids)}")
                        
            except Exception as e:
                UX.warn(f"读取VK已处理文件失败: {e}")
        
        # 清理失败记录，为重新分析做准备
        if failed_ids:
            clean_failed_records(output_path, mapping['comment_id'])
            UX.info(f"清理了 {len(failed_ids)} 个VK失败记录，准备重新分析")

        # 筛选待处理数据：只处理完全未处理的记录 + 失败的记录
        df[mapping['comment_id']] = df[mapping['comment_id']].astype(str)
        unprocessed_ids = set(df[mapping['comment_id']].astype(str)) - (processed_ids - failed_ids)
        df_to_process = df[df[mapping['comment_id']].astype(str).isin(unprocessed_ids)]
        
        if df_to_process.empty:
            UX.ok("所有VK评论已处理")
            return
        
        UX.info(f"待处理: {len(df_to_process)} 条")
        
        # 获取VK处理配置
        vk_config = self.config.get('social_media', {}).get('vk_processing', {})
        
        # 【关键修复】：初始化为PENDING而不是SUCCESS
        df_to_process['processing_status'] = ProcessingStatus.PENDING  # 待处理状态
        
        # 初始化分析列
        analysis_columns = ANALYSIS_COLUMNS + ['Unit_Hash']  # 只添加Unit_Hash字段
        
        df_to_process['Source'] = source
        for col in analysis_columns:
            if col not in df_to_process.columns:
                df_to_process[col] = pd.NA
        
        # 重新排列列顺序，将Unit_Hash放在Post_ID和Source之间
        if 'Unit_Hash' in df_to_process.columns:
            cols = list(df_to_process.columns)
            if 'Post_ID' in cols and 'Source' in cols:
                # 移除Unit_Hash
                cols.remove('Unit_Hash')
                # 找到Post_ID的位置，在其后插入Unit_Hash
                post_idx = cols.index('Post_ID')
                cols.insert(post_idx + 1, 'Unit_Hash')
                df_to_process = df_to_process[cols]
        
        # 使用统一的API配置
        api_strategy = self.config.get('api', {}).get('strategy', {})
        max_concurrent = api_strategy.get('max_concurrent_requests', 2)
        
        # 创建批处理任务（保持原有逻辑）
        batch_tasks = []
        batch_to_comments = {}  # 记录批次对应的评论ID
        
        for post_id, group in df_to_process.groupby(mapping['post_id'], dropna=False):
            if pd.isna(post_id):
                continue
                
            post_text = safe_str_convert(group[mapping['post_text']].iloc[0])
            
            comments_list = []
            comment_ids_in_batch = []
            
            for _, row in group.iterrows():
                comment_id = row[mapping['comment_id']]
                comment_text = row[mapping['comment_text']]
                
                if pd.isna(comment_id) or pd.isna(comment_text) or not str(comment_text).strip():
                    # 空评论直接标记为NO_RELEVANT
                    mask = df_to_process[mapping['comment_id']].astype(str) == str(comment_id)
                    df_to_process.loc[mask, 'processing_status'] = ProcessingStatus.NO_RELEVANT
                    df_to_process.loc[mask, 'relevance'] = '空评论'
                    continue
                    
                comments_list.append({
                    "comment_id": str(comment_id),
                    "comment_text": safe_str_convert(comment_text)
                })
                comment_ids_in_batch.append(str(comment_id))
                
                # 收集基础单元信息
                # 获取channel_name，如果不存在则使用默认值
                channel_name = safe_str_convert(row.get(mapping.get('channel_name', 'channel_name'), '未知频道')) if mapping.get('channel_name', 'channel_name') in df_to_process.columns else '未知频道'
                
                self.Units_collector.append({
                    'Unit_ID': f"VK-{comment_id}",
                    'Source': source,
                    'Post_ID': str(post_id),
                    'Post_Text': post_text,
                    'Comment_Text': safe_str_convert(comment_text),
                    'channel_name': channel_name,
                    'AI_Is_Relevant': None
                })
            
            # 创建批次
            batch_size_limit = vk_config.get('batch_size_limit', 20)
            for i in range(0, len(comments_list), batch_size_limit):
                chunk = comments_list[i:i + batch_size_limit]
                chunk_ids = comment_ids_in_batch[i:i + batch_size_limit]
                if chunk:
                    batch_idx = len(batch_tasks)
                    batch_tasks.append({
                        "post_id": str(post_id),
                        "post_text": post_text,
                        "comments": chunk
                    })
                    batch_to_comments[batch_idx] = chunk_ids
        
        if not batch_tasks:
            UX.warn("没有可处理的批次任务")
            return
            
        UX.info(f"创建了 {len(batch_tasks)} 个批处理任务")
        
        # 处理批次
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            future_to_batch = {
                executor.submit(self._process_batch, batch_tasks[i]): i 
                for i in range(len(batch_tasks))
            }
            progress_bar = tqdm(as_completed(future_to_batch), total=len(batch_tasks), desc="批处理进度")
            
            for future in progress_bar:
                batch_idx = future_to_batch[future]
                expected_comment_ids = batch_to_comments[batch_idx]
                
                try:
                    batch_results = future.result()
                    
                    if batch_results is None:
                        # 整个批次失败
                        UX.warn(f"批次 {batch_idx} 返回None，标记所有评论为失败")
                        for comment_id in expected_comment_ids:
                            mask = df_to_process[mapping['comment_id']].astype(str) == comment_id
                            df_to_process.loc[mask, 'processing_status'] = ProcessingStatus.API_FAILED
                            df_to_process.loc[mask, 'relevance'] = 'API调用失败'
                            for col in analysis_columns:
                                if col != 'Source':
                                    df_to_process.loc[mask, col] = 'API_FAILED'
                        continue
                    
                    if not isinstance(batch_results, list):
                        UX.warn(f"批次 {batch_idx} 返回非列表结果")
                        for comment_id in expected_comment_ids:
                            mask = df_to_process[mapping['comment_id']].astype(str) == comment_id
                            df_to_process.loc[mask, 'processing_status'] = ProcessingStatus.API_FAILED
                            df_to_process.loc[mask, 'relevance'] = '返回格式错误'
                        continue
                    
                    # 处理返回的结果
                    processed_comment_ids = set()
                    
                    for result in batch_results:
                        if not isinstance(result, dict):
                            continue
                            
                        comment_id = str(result.get('comment_id', ''))
                        if not comment_id:
                            continue
                        
                        processed_comment_ids.add(comment_id)
                        mask = df_to_process[mapping['comment_id']].astype(str) == comment_id
                        
                        # 从DataFrame中获取原始comment_text并添加到result中用于哈希计算
                        if mask.any():
                            original_comment_text = df_to_process.loc[mask, mapping['comment_text']].iloc[0]
                            result['comment_text'] = original_comment_text
                        
                        # 【关键】：正确设置processing_status
                        if 'processing_status' in result:
                            df_to_process.loc[mask, 'processing_status'] = result['processing_status']
                        else:
                            # 根据relevance判断
                            if result.get('relevance') == '不相关':
                                df_to_process.loc[mask, 'processing_status'] = ProcessingStatus.NO_RELEVANT
                            elif result.get('relevance') in ['API_FAILED', 'INVALID_RESPONSE', 'EXCEPTION']:
                                df_to_process.loc[mask, 'processing_status'] = ProcessingStatus.API_FAILED
                            else:
                                df_to_process.loc[mask, 'processing_status'] = ProcessingStatus.SUCCESS
                        
                        # 更新Units_collector中的相关性信息
                        for Unit in self.Units_collector:
                            if Unit['Unit_ID'] == f"VK-{comment_id}":
                                Unit['AI_Is_Relevant'] = (result.get('relevance') not in ['不相关', 'API_FAILED', 'INVALID_RESPONSE', 'EXCEPTION'])
                                break
                        
                        # 更新其他字段（包括Unit_Hash）
                        for key, value in result.items():
                            if (key in df_to_process.columns or key == 'Unit_Hash') and key != mapping['comment_id']:
                                if isinstance(value, list):
                                    df_to_process.loc[mask, key] = json.dumps(value, ensure_ascii=False)
                                else:
                                    df_to_process.loc[mask, key] = value
                    
                    # 标记未返回的评论为失败
                    missing_ids = set(expected_comment_ids) - processed_comment_ids
                    if missing_ids:
                        UX.warn(f"批次 {batch_idx} 中有 {len(missing_ids)} 个评论未返回结果，标记为失败")
                        for comment_id in missing_ids:
                            mask = df_to_process[mapping['comment_id']].astype(str) == comment_id
                            df_to_process.loc[mask, 'processing_status'] = ProcessingStatus.API_FAILED
                            df_to_process.loc[mask, 'relevance'] = '未返回结果'
                                            
                    # 定期保存进度
                    save_interval = vk_config.get('save_interval', 5)
                    if (batch_idx + 1) % save_interval == 0:
                        self._save_progress_generic(df_to_process, output_path, mapping['comment_id'])
                        # 显示当前统计
                        success_count = (df_to_process['processing_status'] == ProcessingStatus.SUCCESS).sum()
                        failed_count = (df_to_process['processing_status'] == ProcessingStatus.API_FAILED).sum()
                        no_relevant_count = (df_to_process['processing_status'] == ProcessingStatus.NO_RELEVANT).sum()
                        pending_count = (df_to_process['processing_status'] == ProcessingStatus.PENDING).sum()
                        total_processed = success_count + failed_count + no_relevant_count
                        progress_rate = (total_processed / len(df_to_process)) * 100
                        UX.info(f"📊 VK处理进度 ({progress_rate:.1f}%): 成功{success_count}, 失败{failed_count}, 无相关{no_relevant_count}, 待处理{pending_count}")
                        
                except Exception as e:
                    UX.err(f"处理批次 {batch_idx} 异常: {str(e)[:100]}")
                    # 批次异常，标记所有评论为失败
                    for comment_id in expected_comment_ids:
                        mask = df_to_process[mapping['comment_id']].astype(str) == comment_id
                        df_to_process.loc[mask, 'processing_status'] = ProcessingStatus.API_FAILED
                        df_to_process.loc[mask, 'relevance'] = f'批次异常: {str(e)[:50]}'
                    continue
        
        # 最终检查：将所有仍为PENDING的记录标记为失败
        pending_mask = df_to_process['processing_status'] == ProcessingStatus.PENDING
        if pending_mask.any():
            UX.warn(f"发现 {pending_mask.sum()} 条未处理记录，标记为失败")
            df_to_process.loc[pending_mask, 'processing_status'] = ProcessingStatus.API_FAILED
            df_to_process.loc[pending_mask, 'relevance'] = '未被处理'
        
        # 最终保存
        self._save_progress_generic(df_to_process, output_path, mapping['comment_id'])
        
        # 显示最终统计
        success_count = (df_to_process['processing_status'] == ProcessingStatus.SUCCESS).sum()
        failed_count = (df_to_process['processing_status'] == ProcessingStatus.API_FAILED).sum()
        no_relevant_count = (df_to_process['processing_status'] == ProcessingStatus.NO_RELEVANT).sum()
        
        completion_rate = ((success_count + no_relevant_count) / len(df_to_process)) * 100
        UX.ok(f"📋 VK处理完成总结: 完成度{completion_rate:.1f}% (成功{success_count} + 无相关{no_relevant_count})")
        UX.info(f"   📊 详细统计: 成功{success_count}, 失败{failed_count}, 无相关{no_relevant_count}")
        
        if failed_count > 0:
            UX.warn(f"   ⚠️  仍有{failed_count}条记录处理失败，可再次运行进行智能重试")
    
    def _process_batch(self, batch_task):
        """处理单个批次"""
        try:
            post_text = batch_task['post_text']
            comments_json = json.dumps(batch_task['comments'], ensure_ascii=False)
            
            prompt = self.prompts.VK_BATCH_ANALYSIS.format(
                post_text=str(post_text),
                comments_json=comments_json
            )
            
            # 智能模型选择：根据文本长度选择合适的模型
            combined_text = str(post_text) + " " + comments_json
            text_tokens = count_tokens(combined_text)
            
            # 使用已初始化的阈值
            VK_LONG_TEXT_THRESHOLD = self.thresholds['VK_LONG_TEXT_THRESHOLD']
            
            if text_tokens > VK_LONG_TEXT_THRESHOLD:
                stage_key = 'VK_BATCH_LONG'
                UX.info(f"🔧 VK批次模型选择: ({text_tokens} tokens > {VK_LONG_TEXT_THRESHOLD}) → 长文本模型")
            else:
                stage_key = 'VK_BATCH'
                UX.info(f"🔧 VK批次模型选择: ({text_tokens} tokens) → 标准模型")
            
            # 调用API
            result = self.api_service.call_api_sync(prompt, language='ru', stage_key=stage_key)
            
            # API调用失败
            if result is None:
                UX.warn(f"API调用返回None，批次包含 {len(batch_task['comments'])} 条评论")
                post_id = batch_task.get('post_id')
                return [self._create_failed_record(c['comment_id'], 'API返回None', post_id) 
                        for c in batch_task['comments']]
            
            # 强化API结果类型检查
            if not isinstance(result, list):
                UX.warn(f"批次API返回格式错误（非列表），将整个批次标记为失败。返回内容: {str(result)[:100]}")
                post_id = batch_task.get('post_id')
                return [self._create_failed_record(c['comment_id'], 'API响应格式无效', post_id) 
                        for c in batch_task['comments']]
            
            # 尝试提取结果
            processed_results = []
            
            if isinstance(result, list):
                for item in result:
                    if isinstance(item, dict):
                        # 确保有comment_id
                        if 'comment_id' not in item:
                            continue
                        
                        comment_id = str(item.get('comment_id', ''))
                        
                        # 从batch_task中找到对应的comment_text
                        for comment in batch_task['comments']:
                            if str(comment['comment_id']) == comment_id:
                                item['comment_text'] = comment['comment_text']
                                item['Unit_Text'] = comment['comment_text']  # 添加Unit_Text字段
                                break
                        
                        # 添加processing_status
                        if 'processing_status' not in item:
                            if item.get('relevance') == '不相关':
                                item['processing_status'] = ProcessingStatus.NO_RELEVANT
                            else:
                                item['processing_status'] = ProcessingStatus.SUCCESS
                        
                        self._add_hash_to_record(item, 'comment_text')
                        processed_results.append(item)
                        
            elif isinstance(result, dict):
                # 尝试多个可能的键
                for key in ['analysis', 'results', 'processed_results', 'data', 'comments']:
                    if key in result and isinstance(result[key], list):
                        for item in result[key]:
                            if isinstance(item, dict) and 'comment_id' in item:
                                comment_id = str(item.get('comment_id', ''))
                                
                                if comment_id:
                                    # 从batch_task中找到对应的comment_text
                                    for comment in batch_task['comments']:
                                        if str(comment['comment_id']) == comment_id:
                                            item['comment_text'] = comment['comment_text']
                                            item['Unit_Text'] = comment['comment_text']  # 添加Unit_Text字段
                                            break
                                
                                if 'processing_status' not in item:
                                    if item.get('relevance') == '不相关':
                                        item['processing_status'] = ProcessingStatus.NO_RELEVANT
                                    else:
                                        item['processing_status'] = ProcessingStatus.SUCCESS
                                
                                self._add_hash_to_record(item, 'comment_text')
                                processed_results.append(item)
                        break
            
            # 如果没有提取到有效结果
            if not processed_results:
                UX.warn(f"无法从API响应中提取有效结果，批次包含 {len(batch_task['comments'])} 条评论")
                post_id = batch_task.get('post_id')
                return [self._create_failed_record(c['comment_id'], 'API响应格式无效', post_id) 
                        for c in batch_task['comments']]
            
            # 检查是否所有评论都有结果
            result_ids = {str(r.get('comment_id')) for r in processed_results if r.get('comment_id')}
            expected_ids = {c['comment_id'] for c in batch_task['comments']}
            missing_ids = expected_ids - result_ids
            extra_ids = result_ids - expected_ids
            
            if missing_ids:
                UX.warn(f"API响应缺少 {len(missing_ids)} 条评论的结果")
                post_id = batch_task.get('post_id')
                for comment_id in missing_ids:
                    processed_results.append(self._create_failed_record(comment_id, 'API响应中缺失', post_id))
            
            if extra_ids:
                UX.warn(f"API响应包含 {len(extra_ids)} 条额外评论，将被忽略")
                processed_results = [r for r in processed_results if str(r.get('comment_id', '')) in expected_ids]
                        
            return processed_results
            
        except Exception as e:
            UX.err(f"批次处理异常: {str(e)}")
            post_id = batch_task.get('post_id')
            return [self._create_failed_record(c['comment_id'], f'异常: {str(e)[:50]}', post_id) 
                    for c in batch_task['comments']]
    
    def _create_failed_record(self, comment_id, reason, post_id=None):
        """创建失败记录"""
        record = create_unified_record(ProcessingStatus.API_FAILED, comment_id, 'vk', '', reason)
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
    
    """知乎回答处理器（两步式，与媒体文本对齐）"""
    
    def _get_author_name(self, original_row, mapping):
        """智能获取作者名称：有回答用户名列则使用，无则返回未知作者"""
        author_column = mapping.get("author", "回答用户名")
        if author_column in original_row.index:
            author_value = safe_str_convert(original_row[author_column])
            return author_value if author_value.strip() else '未知作者'
        else:
            return '未知作者'
    
    def _finalize_record(self, result_data, original_row, mapping, answer_id, Unit_index=1):
        """
        将API返回的分析结果封装成一个完整的记录字典。
        """
        if not result_data or not isinstance(result_data, dict):
            return None

        author = self._get_author_name(original_row, mapping)
        
        result_data['speaker'] = author
        result_data['Source'] = '知乎'
        result_data['processing_status'] = ProcessingStatus.SUCCESS
        result_data['Unit_ID'] = f"ZH-{answer_id}-{Unit_index}"
        result_data['Answer_ID'] = f"ZH-{answer_id}"
        result_data['id'] = answer_id  # 兼容旧列名
        result_data['序号'] = answer_id # 兼容旧列名

        # V2格式直接使用，无需转换

        self._add_hash_to_record(result_data, 'Unit_Text')
        
        return result_data
    
    async def process(self, df, output_path, source='知乎'):
        """处理知乎文件"""
        UX.info("处理知乎回答...")
        
        # 获取社交媒体专用的列映射配置
        # 获取知乎列映射配置（统一路径）
        social_config = self.config.get('social_media', {})
        column_mapping = social_config.get('column_mapping', {})
        mapping = column_mapping.get('zhihu', {})
        failed_ids_list = []  # 初始化失败ID列表
        
        # 输入数据模式校验（回答用户名列为可选）
        required_columns = set(mapping.values())
        # 移除可选的回答用户名列
        optional_columns = {mapping.get("author", "回答用户名")}
        required_columns = required_columns - optional_columns
        actual_columns = set(df.columns)
        
        if not required_columns.issubset(actual_columns):
            missing = required_columns - actual_columns
            UX.err(f"知乎文件格式错误，缺少以下必需列: {list(missing)}。已跳过此文件。")
            return failed_ids_list
        
        # 检查可选列是否存在
        author_column = mapping.get("author", "回答用户名")
        if author_column in actual_columns:
            UX.info(f"检测到作者列: {author_column}")
        else:
            UX.info(f"未检测到作者列: {author_column}，将使用'未知作者'")
        
        # 检查已处理的记录（包括成功和失败）
        processed_ids = set()
        failed_ids = set()
        if os.path.exists(output_path):
            try:
                df_existing_check = pd.read_excel(output_path)
                if not df_existing_check.empty and mapping["id"] in df_existing_check.columns:
                    processed_ids, failed_ids = get_processing_state(df_existing_check, mapping["id"])
                    
                    if 'processing_status' in df_existing_check.columns:
                        success_count = (df_existing_check['processing_status'] == ProcessingStatus.SUCCESS).sum()
                        no_relevant_count = (df_existing_check['processing_status'] == ProcessingStatus.NO_RELEVANT).sum()
                        failed_count = (df_existing_check['processing_status'] == ProcessingStatus.API_FAILED).sum()
                        UX.info(f"知乎已处理: {len(processed_ids)} (成功: {success_count}, 无相关: {no_relevant_count})")
                        if failed_ids:
                            UX.info(f"发现知乎失败记录: {len(failed_ids)} 个，将重新分析")
                    else:
                        UX.info(f"知乎已处理: {len(processed_ids)}")
                        
            except Exception as e:
                UX.warn(f"读取知乎已处理文件失败: {e}")
        
        # 清理失败记录，为重新分析做准备
        if failed_ids:
            clean_failed_records(output_path, mapping["id"])
            UX.info(f"清理了 {len(failed_ids)} 个知乎失败记录，准备重新分析")

        # 筛选待处理数据：只处理完全未处理的记录 + 失败的记录
        df[mapping["id"]] = df[mapping["id"]].astype(str)
        unprocessed_ids = set(df[mapping["id"]].astype(str)) - (processed_ids - failed_ids)
        df_to_process = df[df[mapping["id"]].astype(str).isin(unprocessed_ids)]
        
        if df_to_process.empty:
            UX.ok("所有知乎回答已处理")
            return failed_ids_list
        
        UX.info(f"待处理: {len(df_to_process)} 条")
        
        # 并发处理
        api_strategy = self.config.get('api', {}).get('strategy', {})
        max_concurrent = api_strategy.get('max_concurrent_requests', 2)
        semaphore = asyncio.Semaphore(max_concurrent)
        tasks = []
        
        for idx, row in df_to_process.iterrows():
            answer_id = str(row[mapping["id"]])
            tasks.append(self._process_answer(row, mapping, answer_id, semaphore))
        
        all_results = await aio_gather(*tasks)
        
        # 展平结果并收集所有记录（包括成功和失败）
        final_results = []
        for result_item in all_results:
            if isinstance(result_item, tuple) and result_item[0] == 'FAILED':
                # 兼容旧版本的失败处理（不应该再出现）
                failed_ids_list.append(f"ID: {result_item[1]} - Reason: {result_item[2]}")
            elif result_item:  # 这是一个有内容的任务（成功或失败）
                final_results.extend(result_item)
        
        # 保存结果
        if final_results:
            df_results = pd.DataFrame(final_results)
            
            if os.path.exists(output_path):
                df_existing = pd.read_excel(output_path)
                df_final = pd.concat([df_existing, df_results], ignore_index=True)
            else:
                df_final = df_results
            
            df_final.to_excel(output_path, index=False)
            UX.ok(f"保存 {len(final_results)} 条知乎议题单元分析结果")
        
        return failed_ids_list
    
    async def _process_answer(self, row, mapping, answer_id, semaphore):
        """处理单个知乎回答（根据长度智能选择模式）"""
        async with semaphore:
            answer_text = safe_str_convert(row[mapping["answer_text"]])
            question = safe_str_convert(row[mapping["question"]])
            author = self._get_author_name(row, mapping)
            
            if not answer_text.strip():
                return []
            
            try:
                # 根据文本长度选择处理模式（统一使用token计算）
                answer_tokens = count_tokens(answer_text)
                if answer_tokens < self.thresholds['ZHIHU_SHORT_TOKEN_THRESHOLD']:
                    # === 短文本模式：直接分析（类似VK） ===
                    UX.info(f"🔧 知乎回答 {answer_id} 模式选择: ({answer_tokens} tokens) → 短文本直接分析模式")
                    
                    # 使用模板构造提示词
                    prompt = self.prompts.ZHIHU_SHORT_ANALYSIS.format(
                        question=question,
                        answer_text=answer_text
                    )
                    
                    # 智能模型选择：短文本模式直接使用轻量模型
                    stage_key = 'ZHIHU_ANALYSIS_SHORT'
                    UX.info(f"🔧 知乎模型选择: ({answer_tokens} tokens < {self.thresholds['ZHIHU_SHORT_TOKEN_THRESHOLD']}) → 轻量模型")
                    
                    result = await self.api_service.call_api_async(
                        prompt, 'zh', stage_key
                    )
                    
                    if result:
                        result['Unit_Text'] = answer_text # 确保Unit_Text存在
                        final_record = self._finalize_record(result, row, mapping, answer_id)
                        
                        # 添加到Units_collector
                        self.Units_collector.append({
                            'Unit_ID': f"ZH-{answer_id}-1",
                            'Source': '知乎',
                            'Question': question,
                            'Answer_Text': answer_text[:500],
                            'Author': author,
                            'AI_Is_Relevant': True
                        })
                        
                        return [final_record] if final_record else None
                    else:
                        # API失败，返回统一失败记录
                        failed_record = create_unified_record(ProcessingStatus.API_FAILED, answer_id, '知乎', answer_text[:200], '知乎短文本分析失败')
                        failed_record['Unit_Text'] = f'[分析失败] {answer_text[:100]}...'
                        return [failed_record]
                        
                else:
                    # === 长文本模式：两步式分析 ===
                    UX.info(f"🔧 知乎回答 {answer_id} 模式选择: ({answer_tokens} tokens) → 两步式分析模式")
                    
                    # 第一步：议题单元划分
                    prompt1 = self.prompts.ZHIHU_CHUNKING.format(full_text=answer_text)
                    result1 = await self.api_service.call_api_async(prompt1, 'zh', 'ZHIHU_CHUNKING')
                    
                    if not result1:
                        failed_record = create_unified_record(ProcessingStatus.API_FAILED, answer_id, '知乎', '', '知乎切分失败')
                        failed_record['Unit_Text'] = f'[切分失败] {answer_text[:100]}...'
                        return [failed_record]
                    
                    chapters = result1.get('argument_chapters', [])
                    if not chapters:
                        chapters = [{'Unit_Text': answer_text}]
                    
                    # 添加到Units_collector
                    self.Units_collector.append({
                        'Unit_ID': f"ZH-{answer_id}",
                        'Source': '知乎',
                        'Question': question,
                        'Answer_Text': answer_text[:500],  # 截取预览
                        'Author': author,
                        'AI_Is_Relevant': None  # 后续更新
                    })
                    
                    # 第二步：对每个议题单元进行分析
                    results = []
                    for i, chapter in enumerate(chapters):
                        Unit_Text = chapter.get('Unit_Text', '')
                        if not Unit_Text.strip():
                            continue
                        
                        prompt2 = self.prompts.ZHIHU_ANALYSIS.format(
                            question=question,
                            Unit_Text=Unit_Text
                        )
                        
                        # 智能模型选择：根据议题单元长度选择合适的模型
                        unit_tokens = count_tokens(Unit_Text)
                        if unit_tokens < self.thresholds['ZHIHU_SHORT_TOKEN_THRESHOLD']:
                            stage_key = 'ZHIHU_ANALYSIS_SHORT'
                            UX.info(f"🔧 知乎议题单元模型选择: ({unit_tokens} tokens < {self.thresholds['ZHIHU_SHORT_TOKEN_THRESHOLD']}) → 轻量模型")
                        elif unit_tokens > self.thresholds['ZHIHU_LONG_TOKEN_THRESHOLD']:
                            stage_key = 'ZHIHU_ANALYSIS_LONG'
                            UX.info(f"🔧 知乎议题单元模型选择: ({unit_tokens} tokens > {self.thresholds['ZHIHU_LONG_TOKEN_THRESHOLD']}) → 高性能模型")
                        else:
                            stage_key = 'ZHIHU_ANALYSIS'
                            UX.info(f"🔧 知乎议题单元模型选择: ({unit_tokens} tokens) → 标准模型")
                        
                        result2 = await self.api_service.call_api_async(prompt2, 'zh', stage_key)
                        
                        if result2:
                            # 构建完整记录
                            Unit_record = {
                                "Unit_Text": Unit_Text,
                                "expansion_logic": f"第{i+1}个论证章节",  
                                **result2  # 包含所有分析维度
                            }
                            
                            final_record = self._finalize_record(Unit_record, row, mapping, answer_id, Unit_index=i + 1)
                            if final_record:
                                results.append(final_record)
                    
                    # 更新Units_collector中的相关性
                    for Unit in self.Units_collector:
                        if Unit['Unit_ID'] == f"ZH-{answer_id}":
                            Unit['AI_Is_Relevant'] = bool(results)
                            break
                    
                    if results:
                        return results
                    else:
                        # 长文本分析失败，返回统一失败记录
                        failed_record = create_unified_record(ProcessingStatus.API_FAILED, answer_id, '知乎', answer_text[:200], '知乎长文本分析失败')
                        failed_record['Unit_Text'] = f'[分析失败] {answer_text[:100]}...'
                        return [failed_record]
                    
            except Exception as e:
                UX.warn(f"处理回答 {answer_id} 失败: {str(e)[:100]}")
                # 返回统一失败记录
                failed_record = create_unified_record(ProcessingStatus.API_FAILED, answer_id, '知乎', answer_text[:200], f'异常: {str(e)[:50]}')
                failed_record['Unit_Text'] = f'[异常] {str(e)[:100]}'
                return [failed_record]

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
                    ProcessingStatus.API_FAILED, original_id, source, 
                    full_text[:200], f"文本过长({text_tokens} tokens)"
                )
                return original_id, [failed_record]
            
            UX.info(f"🚀 ID {original_id}: 开始两阶段处理 ({text_tokens} tokens, {language})")

            # 第一阶段：议题单元提取
            UX.info(f"📋 ID {original_id}: [1/2] 议题单元提取阶段 - 使用模型: {self.get_stage_model('UNIT_EXTRACTION')}")

            extraction_prompt = self.prompts.UNIT_EXTRACTION.replace("{full_text}", full_text)
            
            extraction_result = await self.api_service.get_analysis(
                extraction_prompt, 'analyzed_Units', language,
                model_name=self.get_stage_model('UNIT_EXTRACTION'), 
                stage_key='UNIT_EXTRACTION',
                context_label=f"{original_id}:EXTRACTION"
            )
            
            if extraction_result is None:
                UX.warn(f"❌ ID {original_id}: [1/2] 第一阶段议题单元提取失败")
                failed_record = create_unified_record(
                    ProcessingStatus.API_FAILED, original_id, source, 
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
                    model_name=self.get_stage_model('UNIT_ANALYSIS'), 
                    stage_key='UNIT_ANALYSIS',
                    context_label=f"{unit_id}:ANALYSIS"
                )
                
                if analysis_result is None:
                    UX.warn(f"❌ {unit_id}: 第二阶段分析失败")
                    failed_unit = {
                        "processing_status": ProcessingStatus.API_FAILED,
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
                ProcessingStatus.API_FAILED, original_id, source, 
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