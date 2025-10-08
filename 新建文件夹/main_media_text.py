# -*- coding: utf-8 -*-
"""
媒体文本分析主入口 - 种子扩展两阶段分析
"""
import os
import glob
import asyncio
import aiohttp
import pandas as pd
from asyncio import Lock
from tqdm.asyncio import tqdm as aio_tqdm

# 导入核心模块
from core import (
    UX,
    load_config,
    APIService,
    ReliabilityTestModule,
    safe_str_convert,
    identify_source,
    create_auto_retry_manager,
)
from core.media_text_processor import MediaTextProcessor
from core.utils import ProcessingStatus, get_processing_state

# 加载配置
CONFIG = load_config()
if CONFIG is None:
    raise RuntimeError("无法加载配置文件，程序退出")

# 新的结构化配置读取
# API配置
api_config = CONFIG['api']
api_strategy = api_config['strategy']
MAX_CONCURRENT_REQUESTS = api_strategy['max_concurrent_requests']
QUEUE_TIMEOUT = api_strategy['queue_timeout_sec']

# 媒体文本流程配置
media_config = CONFIG.get('media_text', {})

# 输入输出路径配置
INPUT_PATH = media_config['paths']['input']
OUTPUT_PATH = media_config['paths']['output']

# 通用自动重试配置
AUTO_RETRY_CONFIG = CONFIG['processing']['auto_retry']
AUTO_RETRY_ENABLED = AUTO_RETRY_CONFIG['enabled']
MAX_RETRY_ROUNDS = AUTO_RETRY_CONFIG['max_rounds']
RETRY_DELAY_MINUTES = AUTO_RETRY_CONFIG['delay_minutes']
MIN_FAILED_THRESHOLD = AUTO_RETRY_CONFIG['min_failed_threshold']

# 信度检验配置
RELIABILITY_TEST_CONFIG = CONFIG['reliability_test']
RELIABILITY_TEST_MODE = RELIABILITY_TEST_CONFIG['enabled']
RELIABILITY_SAMPLING_CONFIG = RELIABILITY_TEST_CONFIG['sampling_config']

# 列映射和输出列配置
COLUMN_MAPPING = CONFIG.get('media_text', {}).get('column_mapping', {})
REQUIRED_OUTPUT_COLUMNS = CONFIG.get('media_text', {}).get('required_output_columns', CONFIG.get('required_output_columns', []))

# 模型池配置（全局统一）
MODEL_POOLS = CONFIG.get('model_pools', {})
API_CONFIG = api_config

# 全局锁
file_write_lock = Lock()

# 提示词类
class Prompts:
    def __init__(self):
        self.prompts_dir = os.path.join(os.path.dirname(__file__), 'prompts')
        self._load_prompts()

    def _load_prompts(self):
        prompt_files = {
            'UNIT_EXTRACTION': 'SEED.txt',
            'UNIT_ANALYSIS': 'ANALYSIS.txt'
        }
        for attr_name, filename in prompt_files.items():
            file_path = os.path.join(self.prompts_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    setattr(self, attr_name, f.read())
            except Exception as e:
                UX.err(f"加载提示词文件 {filename} 失败: {e}")
                setattr(self, attr_name, "")

# 创建提示词实例
prompts = Prompts()

async def api_worker(name, task_queue, results_queue, api_service, source, pbar, processor, output_file_path=None, failed_unit_ids=None, is_unit_mode=False):
   while True:
       try:
           item = await task_queue.get()
           if item is None:
               break
           
           # 每10个任务检查一次活动状态
           if pbar.n % 10 == 0:
               UX.check_activity()
           
           try:
               item_with_source = (item[0], item[1], source)
               current_id = safe_str_convert(item[1].get(COLUMN_MAPPING["ID"], "unknown")) if len(item) > 1 else "unknown"
               original_id, result_Units = await processor.process_row(
                   item_with_source,
                   output_file_path=output_file_path,
                   failed_unit_ids=failed_unit_ids
               )
               await results_queue.put((original_id, result_Units, None))  # 两阶段模型：只返回议题单元
           except Exception as e:
               error_brief = str(e)[:30] + "..." if len(str(e)) > 30 else str(e)
               UX.api_failed(f"Worker-{name}", error_brief)
               original_id = safe_str_convert(item[1].get(COLUMN_MAPPING["ID"], "unknown")) if len(item) > 1 else "unknown"
               from core.utils import create_unified_record
               await results_queue.put((original_id, [create_unified_record(ProcessingStatus.STAGE_2_FAILED, original_id, source, "", f"Worker错误: {str(e)[:100]}")], None))
           finally:
               # 保证无论成功失败，任务完成后都更新进度条
               # 单元续传模式：按当前文章的失败单元数更新；普通模式：按文章数更新
               if is_unit_mode and failed_unit_ids and current_id in failed_unit_ids:
                   pbar.update(len(failed_unit_ids[current_id]))
               else:
                   pbar.update(1)
               task_queue.task_done()
       
       except asyncio.CancelledError:
           break
       except Exception as e:
           error_brief = str(e)[:30] + "..." if len(str(e)) > 30 else str(e)
           UX.api_failed(f"Worker-{name}-Critical", error_brief)

async def saver_worker(results_queue, df_input, output_file_path, total_tasks=None):
    """
    [两阶段模型] 批量保存worker：
    - 缓冲并分批保存议题单元分析结果
    """
    analysis_buffer = []
    received_count = 0
    
    # 从配置读取缓冲区大小阈值
    general_config = CONFIG.get('processing', {}).get('general', {})
    ANALYSIS_BUFFER_LIMIT = general_config.get('buffer_limit', 20)    # 分析结果缓冲区

    async def save_analysis_batch(data):
        if not data:
            return
        async with file_write_lock:
            # 调用现有的分析结果保存函数
            save_data_to_excel(data, df_input, output_file_path)

    while True:
        try:
            # 等待结果，从配置读取超时时间
            item = await asyncio.wait_for(results_queue.get(), timeout=QUEUE_TIMEOUT)
            if item is None: # 收到结束信号
                await save_analysis_batch(analysis_buffer)
                break

            original_id, result_Units, _ = item  # 第三个参数为兼容性保留，两阶段模型中未使用
            received_count += 1

            # 收集分析结果到缓冲区
            if result_Units:
                for unit in result_Units:
                    unit[COLUMN_MAPPING["ID"]] = original_id
                analysis_buffer.extend(result_Units)

            # 检查是否需要清空并保存缓冲区
            if len(analysis_buffer) >= ANALYSIS_BUFFER_LIMIT:
                await save_analysis_batch(analysis_buffer)
                analysis_buffer = []

            results_queue.task_done()
            if total_tasks and received_count >= total_tasks:
                await save_analysis_batch(analysis_buffer)
                break

        except asyncio.TimeoutError:
            # 超时后保存所有缓冲区内容，防止在任务间隙丢失数据
            await save_analysis_batch(analysis_buffer)
            analysis_buffer = []
            if total_tasks and received_count >= total_tasks:
                break
            continue

        except asyncio.CancelledError:
            await save_analysis_batch(analysis_buffer)
            break

        except Exception as e:
            UX.err(f"保存线程错误: {e}")

def save_data_to_excel(new_Units_list, df_input, output_file_path):
   try:
       df_existing = pd.DataFrame()
       if os.path.exists(output_file_path):
           try:
               df_existing = pd.read_excel(output_file_path)
           except Exception as e:
               UX.warn(f"读取现有文件失败: {e}")
       df_new_Units = pd.DataFrame(new_Units_list)

       if df_new_Units.empty:
           return

       df_existing = ensure_required_columns(df_existing)
       df_new_Units = ensure_required_columns(df_new_Units)
       
       if COLUMN_MAPPING["ID"] in df_input.columns and COLUMN_MAPPING["ID"] in df_new_Units.columns:
           df_input[COLUMN_MAPPING["ID"]] = df_input[COLUMN_MAPPING["ID"]].astype(str)
           df_new_Units[COLUMN_MAPPING["ID"]] = df_new_Units[COLUMN_MAPPING["ID"]].astype(str)
           new_original_ids = df_new_Units[COLUMN_MAPPING["ID"]].unique()
           
           # 🔧 修复核心问题：清理旧的"概要级"失败记录
           # 当一篇文章被重新处理成功时，删除它的旧失败记录
           if not df_existing.empty and 'Unit_ID' in df_existing.columns:
               # 识别"概要级"记录：Unit_ID不包含"-Unit-"的记录
               # 例如："(3)-RU1012-API_FAILED" 或 "(3)-RU1012-NO_RELEVANT"
               summary_mask = (
                   df_existing[COLUMN_MAPPING["ID"]].isin(new_original_ids) & 
                   ~df_existing['Unit_ID'].str.contains('-Unit-', na=False)
               )
               
               removed_count = summary_mask.sum()
               if removed_count > 0:
                   df_existing = df_existing[~summary_mask].copy()

           input_base_cols = [col for col in df_input.columns if col in df_new_Units.columns and col != COLUMN_MAPPING["ID"]]
           if input_base_cols:
               df_new_Units = df_new_Units.drop(columns=input_base_cols)
           
           df_input_subset = df_input[df_input[COLUMN_MAPPING["ID"]].isin(new_original_ids)].copy()
           df_newly_merged = pd.merge(df_input_subset, df_new_Units, on=COLUMN_MAPPING["ID"], how='left')
       else:
           df_newly_merged = df_new_Units.copy()

       df_final_to_save = pd.concat([df_existing, df_newly_merged], ignore_index=True)

       if 'Unit_ID' in df_final_to_save.columns:
           df_final_to_save = df_final_to_save.drop_duplicates(subset=['Unit_ID'], keep='last')

       df_final_to_save = reorder_columns(df_final_to_save, list(df_input.columns))
       df_final_to_save.to_excel(output_file_path, index=False)
       UX.ok(f"已保存: {os.path.basename(output_file_path)} (累计 {len(df_final_to_save)} 条记录)")

   except Exception as e:
       UX.err(f"Excel保存失败: {e}")

async def main_async():
   UX.start_run()
   UX.phase("长文本分析器启动")
   UX.info(f"信度检验模式: {'开启' if RELIABILITY_TEST_MODE else '关闭'}")
   
   # 启动活动监控任务（覆盖整个处理流程）
   monitor_task = asyncio.create_task(activity_monitor())
   
   try:
       # 创建自动重试管理器
       retry_manager = create_auto_retry_manager(CONFIG, OUTPUT_PATH)
       
       # 使用自动重试管理器执行主处理逻辑
       await retry_manager.run_with_auto_retry(main_processing_logic)
   finally:
       # 取消活动监控任务
       monitor_task.cancel()
       try:
           await monitor_task
       except asyncio.CancelledError:
           pass

async def activity_monitor():
    """活动监控任务，每5分钟检查一次"""
    while True:
        await asyncio.sleep(300)  # 5分钟
        UX.check_activity()

async def main_processing_logic():
    """主处理逻辑（从原main_async函数移动而来）"""
    
    # 活动监控任务现在在main_async中管理，无需在此创建
    
    # 验证两阶段模型配置
    required_keys = ["media_text_extraction", "media_text_analysis"]
    missing = [k for k in required_keys if k not in MODEL_POOLS]
    if missing:
        raise ValueError(f"缺少两阶段模型池配置: {missing}")

    # 检查API密钥
    api_keys = CONFIG['api']['credentials']['keys']
    if not api_keys or not api_keys[0]:
        UX.err("未提供有效API密钥")
        return

    # 确定输入文件
    if os.path.isdir(INPUT_PATH):
        files_to_process = glob.glob(os.path.join(INPUT_PATH, '*.xlsx'))
        is_folder_mode = True
        os.makedirs(OUTPUT_PATH, exist_ok=True)
    elif os.path.isfile(INPUT_PATH):
        files_to_process = [INPUT_PATH]
        is_folder_mode = False
    else:
        UX.err("输入路径无效")
        return

    async with aiohttp.ClientSession() as session:
        api_service = APIService(session, CONFIG)
        # 创建媒体文本处理器
        processor = MediaTextProcessor(api_service, CONFIG, prompts)

        for file_path in files_to_process:
            file_basename = os.path.basename(file_path)
            
            # 文件处理开始标记
            UX.phase(f"处理文件: {file_basename}")

            # 识别信源
            source = identify_source(file_basename)
            UX.info(f"信源: {source}")

            output_file_path = os.path.join(OUTPUT_PATH, f"(不能删)analyzed_{file_basename}") if is_folder_mode else OUTPUT_PATH

            # 读取输入文件
            try:
                df_input = pd.read_excel(file_path)
                if COLUMN_MAPPING.get("MEDIA_TEXT", "text") not in df_input.columns:
                    UX.err("文件缺少必要的文本列")
                    continue
            except Exception as e:
                UX.err(f"读取文件失败: {e}")
                continue

            # 两阶段模型：直接处理议题单元，无需中间数据库

            # 🔍 构建断点续传计划
            UX.resume_plan(file_basename)
            total_input_articles = len(set(df_input[COLUMN_MAPPING["ID"]].astype(str)))
            
            # 使用通用方法获取断点续传信息
            never_processed_ids = processor.get_never_processed_ids(
                output_file_path, df_input, COLUMN_MAPPING["ID"]
            )
            failed_stage1_ids = processor.load_failed_stage1_ids(
                output_file_path, COLUMN_MAPPING["ID"]
            )
            failed_units_raw = processor.load_failed_units(output_file_path)
            
            # 构建failed_unit_ids字典（按文章ID分组）
            failed_unit_ids = {}
            for unit in failed_units_raw:
                article_id = str(unit.get(COLUMN_MAPPING["ID"], ''))
                unit_id = str(unit.get('Unit_ID', ''))
                if article_id and unit_id:
                    failed_unit_ids.setdefault(article_id, set()).add(unit_id)
            
            # 合并第一阶段失败的ID到never_processed
            never_processed_ids = never_processed_ids | failed_stage1_ids
            
            ids_to_process = set(never_processed_ids) | set(failed_unit_ids.keys())
            failed_units_count = sum(len(v) for v in failed_unit_ids.values())
            
            # 计算已完成单元数
            completed_units = 0
            if os.path.exists(output_file_path):
                try:
                    df_existing = pd.read_excel(output_file_path)
                    if 'processing_status' in df_existing.columns:
                        completed_units = (df_existing['processing_status'] == ProcessingStatus.SUCCESS).sum()
                except:
                    pass

            # 清晰区分文章级和单元级统计
            UX.info(f"总文章数: {total_input_articles} | 已完成单元: {completed_units}")
            
            # 互斥统计：需要阶段1的文章 vs 只需阶段2的失败单元
            stage1_articles = len(never_processed_ids)  # 需要切分的文章
            retry_units = failed_units_count  # 只需重新分析的失败单元
            
            if stage1_articles > 0 and retry_units > 0:
                UX.info(f"本次处理: {stage1_articles} 篇文章（需切分） + {retry_units} 个失败单元（仅分析）")
            elif stage1_articles > 0:
                UX.info(f"本次处理: {stage1_articles} 篇文章")
            elif retry_units > 0:
                UX.info(f"本次处理: {retry_units} 个失败单元")
            
            UX.resume_end()

            df_to_process = df_input[df_input[COLUMN_MAPPING["ID"]].astype(str).isin(ids_to_process)].copy()

            if len(df_to_process) > 0:
                UX.phase(f"两阶段处理: 单元提取 → 深度分析")

                # 创建任务队列
                task_queue = asyncio.Queue()
                results_queue = asyncio.Queue()
                total_tasks = len(df_to_process)

                # 添加任务到队列
                for item in df_to_process.iterrows():
                    await task_queue.put(item)

                # 创建进度条 - 智能选择计数单位
                is_unit_retry_only = len(never_processed_ids) == 0 and failed_units_count > 0
                if is_unit_retry_only:
                    # 纯单元续传：按单元数计数
                    progress_total = failed_units_count
                    progress_desc = f"续传 {failed_units_count} 个失败单元"
                else:
                    # 包含新文章：按文章数计数
                    progress_total = total_tasks
                    progress_desc = f"处理 {len(df_to_process)} 篇文章"
                
                pbar = aio_tqdm(total=progress_total, desc=progress_desc, 
                               bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

                # 创建保存任务 
                saver_task = asyncio.create_task(
                    saver_worker(results_queue, df_input, output_file_path, total_tasks)
                )

                # 创建工作任务
                worker_tasks = [
                    asyncio.create_task(
                        api_worker(
                            f'worker-{i}',
                            task_queue,
                            results_queue,
                            api_service,
                            source,
                            pbar,
                            processor,
                            output_file_path=output_file_path,
                            failed_unit_ids=failed_unit_ids,
                            is_unit_mode=is_unit_retry_only
                        )
                    )
                    for i in range(MAX_CONCURRENT_REQUESTS)
                ]

                # 等待所有任务都被worker处理完毕
                await task_queue.join()
                
                # 向所有worker发送结束信号
                for _ in range(MAX_CONCURRENT_REQUESTS):
                    await task_queue.put(None)
                
                # 等待所有worker正确关闭
                await asyncio.gather(*worker_tasks)
                
                # 向saver发送结束信号并等待保存完成
                await results_queue.put(None)
                await saver_task
                
                pbar.close()
            
            elif len(ids_to_process) == 0:
                UX.ok(f"文件 {file_basename} | 所有文章已处理完毕")
                continue
       
            # 🎉 处理完成总结
            try:
                df_final_check = pd.read_excel(output_file_path)
                if not df_final_check.empty and 'processing_status' in df_final_check.columns:
                    final_success = (df_final_check['processing_status'] == ProcessingStatus.SUCCESS).sum()
                    final_no_relevant = (df_final_check['processing_status'] == ProcessingStatus.NO_RELEVANT).sum()
                    final_failed = df_final_check['processing_status'].isin([ProcessingStatus.STAGE_1_FAILED, ProcessingStatus.STAGE_2_FAILED]).sum()
                    final_total_units = len(df_final_check)
                    
                    # 计算文章级别完成度
                    final_processed_ids, final_failed_ids = get_processing_state(df_final_check, COLUMN_MAPPING["ID"])
                    final_completed_articles = len(final_processed_ids - final_failed_ids)
                    final_completion_rate = (final_completed_articles / max(1, total_input_articles)) * 100
                    
                    # 文件处理结果总结
                    UX.phase(f"文件 {file_basename} 处理完成")
                    UX.ok(f"文章统计 | 总计 {total_input_articles} | 已完成 {final_completed_articles} | 完成率 {final_completion_rate:.1f}%")
                    UX.ok(f"单元统计 | 总计 {final_total_units} | 成功 {final_success} | 失败 {final_failed} | 无相关 {final_no_relevant}")
                    
                    if final_failed > 0:
                        UX.warn(f"  失败 {final_failed} 个单元，可再次运行重试")
                    else:
                        UX.ok(f"  所有单元均已成功处理")
                else:
                    UX.ok(f"文件 {file_basename} 处理完成")
            except Exception:
                UX.ok(f"文件 {file_basename} 处理完成")

    # 生成信度检验文件
    if RELIABILITY_TEST_MODE:
        UX.phase("生成信度检验文件")
        import traceback
        try:
            result_files = glob.glob(os.path.join(OUTPUT_PATH, "*analyzed_*.xlsx"))
            input_files = glob.glob(os.path.join(INPUT_PATH, "*.xlsx"))

            if not result_files or not input_files:
                UX.warn("未找到信度检验所需的输入/结果文件")
            else:
                result_frames = []
                for path in result_files:
                    df_res = pd.read_excel(path)
                    # 不覆盖Excel里已有的Source，信度检验使用原始Source值
                    # source_name = os.path.basename(path).replace('(不能删)analyzed_', '')
                    # df_res['Source'] = identify_source(source_name)
                    result_frames.append(df_res)
                df_all_results = pd.concat(result_frames, ignore_index=True)

                input_frames = []
                for path in input_files:
                    df_in = pd.read_excel(path)
                    # 根据文件名识别并添加Source列（媒体文本输入文件不含Source）
                    source_name = identify_source(os.path.basename(path))
                    df_in['Source'] = source_name
                    input_frames.append(df_in)
                df_all_input = pd.concat(input_frames, ignore_index=True)

                from core.reliability import create_reliability_test_module

                reliability_module = create_reliability_test_module(
                    output_path=OUTPUT_PATH,
                    sampling_config=RELIABILITY_SAMPLING_CONFIG,
                    random_seed=CONFIG.get('project', {}).get('random_seed', 42)
                )

                reliability_module.generate_files(df_all_results, df_all_input)
                UX.ok("信度检验文件生成完成")
        except Exception as e:
            UX.err(f"信度检验文件生成失败: {e}")
            UX.err(traceback.format_exc())
    
    UX.phase("所有任务完成")

# 添加缺失的辅助函数
def ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    """确保结果表具备统一列：缺失则以空值补齐，不移除已有列。"""
    if df is None or df.empty:
        return pd.DataFrame(columns=REQUIRED_OUTPUT_COLUMNS)
    
    for col in REQUIRED_OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df

def reorder_columns(df: pd.DataFrame, df_input_columns: list) -> pd.DataFrame:
    """将列顺序整理为：输入表原有列在前 + REQUIRED_OUTPUT_COLUMNS（去重保序）"""
    if df is None or df.empty:
        return df
    
    preferred = list(dict.fromkeys(list(df_input_columns) + REQUIRED_OUTPUT_COLUMNS))
    # 保留表内已有的列，且按照 preferred 顺序重排
    ordered_existing = [c for c in preferred if c in df.columns]
    # 追加任何未在 preferred 中但存在于 df 的列，避免信息丢失
    tail = [c for c in df.columns if c not in preferred]
    return df.reindex(columns=ordered_existing + tail)

# 程序入口
if __name__ == "__main__":
    import sys
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main_async())