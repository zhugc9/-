# -*- coding: utf-8 -*-
"""
社交媒体分析主入口 - VK和知乎数据处理
"""
import os
import asyncio
import aiohttp
import pandas as pd
import glob
import shutil

# 导入核心模块
from core import (
    UX,
    load_config,
    APIService,
    identify_source,
    count_tokens,
    create_auto_retry_manager,
    VKProcessor,
    ZhihuProcessor,
)
from core.utils import ProcessingStatus

# 加载配置
CONFIG = load_config()
if CONFIG is None:
    raise RuntimeError("无法加载配置文件，程序退出")

# 新的结构化配置读取
# API配置
api_config = CONFIG['api']
api_strategy = api_config['strategy']
MAX_CONCURRENT_REQUESTS = api_strategy['max_concurrent_requests']
API_RETRY_ATTEMPTS = api_strategy['attempts_per_model']
RATE_LIMIT_BASE_DELAY = api_strategy['retry_delays_sec'][0]  # 使用第一个延迟值

# 社交媒体流程配置
social_config = CONFIG.get('social_media', {})
INPUT_PATH = social_config['paths']['input']
OUTPUT_PATH = social_config['paths']['output']
text_thresholds = social_config['text_length_thresholds']
VK_LONG_TEXT_THRESHOLD = text_thresholds['vk_long_text']
ZHIHU_SHORT_TOKEN_THRESHOLD = text_thresholds['zhihu_short_text']
ZHIHU_LONG_TOKEN_THRESHOLD = text_thresholds['zhihu_long_text']

# 通用处理配置
general_config = CONFIG['processing']['general']
SKIP_FAILED_TEXTS = general_config['skip_on_api_failure']

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

# 列映射配置
COLUMN_MAPPING = social_config.get('column_mapping', {})

# 全局模型池配置
MODEL_POOLS = CONFIG.get('model_pools', {})

# 提示词类（从外部文件加载）
class Prompts:
    def __init__(self):
        self.prompts_dir = os.path.join(os.path.dirname(__file__), 'prompts')
        self._load_prompts()
    
    def _load_prompts(self):
        prompt_files = {
            'VK_BATCH_ANALYSIS': 'VK_BATCH_ANALYSIS.txt',
            'ZHIHU_CHUNKING': 'ZHIHU_CHUNKING.txt', 
            'ZHIHU_ANALYSIS': 'ZHIHU_ANALYSIS.txt',
            'ZHIHU_SHORT_ANALYSIS': 'ZHIHU_SHORT_ANALYSIS.txt'
        }
        
        for attr_name, filename in prompt_files.items():
            file_path = os.path.join(self.prompts_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                setattr(self, attr_name, content)
            except FileNotFoundError:
                UX.warn(f"提示词文件 {file_path} 不存在，使用默认内容")
                setattr(self, attr_name, f"# 提示词文件 {filename} 未找到")
            except Exception as e:
                UX.warn(f"加载提示词文件 {file_path} 失败: {e}")
                setattr(self, attr_name, f"# 提示词文件 {filename} 加载失败: {e}")

# 创建全局提示词实例
prompts = Prompts()

async def main_processing_logic():
    """主处理逻辑（可被自动重试调用）"""
    UX.phase("社交媒体分析器启动")
    UX.info(f"信度检验模式: {'开启' if RELIABILITY_TEST_MODE else '关闭'}")
    UX.info(f"配置: 并发{MAX_CONCURRENT_REQUESTS} | VK阈值{VK_LONG_TEXT_THRESHOLD} | 知乎短{ZHIHU_SHORT_TOKEN_THRESHOLD}/长{ZHIHU_LONG_TOKEN_THRESHOLD} tokens")
    
    # 检查配置
    required_models = [
        "vk_batch_standard",
        "vk_batch_long",
        "ZHIHU_CHUNKING",
        "ZHIHU_ANALYSIS_SHORT",
        "ZHIHU_ANALYSIS",
        "ZHIHU_ANALYSIS_LONG"
    ]
    missing = [m for m in required_models if m not in MODEL_POOLS]
    if missing:
        UX.err(f"缺少模型配置: {missing}")
        return
    
    # 创建输出目录
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # 获取输入文件
    files = [f for f in os.listdir(INPUT_PATH) 
            if f.endswith('.xlsx') and not f.startswith('~$')]
    
    if not files:
        UX.warn("未找到输入文件")
        return
    
    all_Units_data = []
    all_results_files = []
    
    # 分类文件
    vk_files = []
    zhihu_files = []
    
    for f in files:
        source = identify_source(f)
        if source == 'vk':
            vk_files.append(f)
        elif source == '知乎':
            zhihu_files.append(f)
        else:
            UX.warn(f"无法识别文件类型: {f}")
    
    failed_zhihu_ids = []

    async with aiohttp.ClientSession() as session:
        api_service = APIService(session, CONFIG)

        # 处理VK文件
        if vk_files:
            UX.phase("处理VK文件")

            for filename in vk_files:
                input_file = os.path.join(INPUT_PATH, filename)
                output_file = os.path.join(OUTPUT_PATH, f"(不能删)analyzed_{filename}")

                try:
                    df = pd.read_excel(input_file)
                    processor = VKProcessor(api_service, CONFIG, prompts)
                    await processor.process(df, output_file, 'vk')

                    if processor.Units_collector:
                        all_Units_data.extend(processor.Units_collector)
                    all_results_files.append(output_file)

                except Exception as e:
                    UX.err(f"处理VK文件 {filename} 失败: {e}")

        # 处理知乎文件
        if zhihu_files:
            UX.phase("处理知乎文件")

            for filename in zhihu_files:
                input_file = os.path.join(INPUT_PATH, filename)
                output_file = os.path.join(OUTPUT_PATH, f"(不能删)analyzed_{filename}")

                try:
                    df = pd.read_excel(input_file)
                    processor = ZhihuProcessor(api_service, CONFIG, prompts)
                    failed_ids_in_file = await processor.process(df, output_file, '知乎')
                    if failed_ids_in_file:
                        failed_zhihu_ids.extend(failed_ids_in_file)

                    if processor.Units_collector:
                        all_Units_data.extend(processor.Units_collector)
                    all_results_files.append(output_file)

                except Exception as e:
                    UX.err(f"处理知乎文件 {filename} 失败: {e}")
                    failed_zhihu_ids.append(f"FILE_FAILED: {filename} - {str(e)[:100]}")

        api_service.print_statistics()

    if failed_zhihu_ids:
        log_path = os.path.join(OUTPUT_PATH, 'zhihu_failed_ids_log.txt')
        try:
            with open(log_path, 'w', encoding='utf-8') as f:
                for failed_id in failed_zhihu_ids:
                    f.write(f"{failed_id}\n")
            UX.warn(f"知乎失败任务已记录到: {log_path}")
        except Exception as e:
            UX.warn(f"写入知乎失败日志失败: {e}")
    # 生成信度检验文件
    if RELIABILITY_TEST_MODE:
        UX.phase("生成信度检验文件")
        
        # 调试信息
        UX.info(f"数据收集状态 - 结果文件: {len(all_results_files)}个")
        
        # 合并所有结果
        final_path = None
        if all_results_files:
            all_results = []
            for file in all_results_files:
                if os.path.exists(file):
                    df = pd.read_excel(file)
                    all_results.append(df)
                    filename = os.path.basename(file)
                    UX.info(f"加载 {filename} 用于信度检验，{len(df)} 条记录")
                else:
                    UX.warn(f"结果文件不存在: {file}")
            
            if all_results:
                df_final = pd.concat(all_results, ignore_index=True)
                final_path = os.path.join(OUTPUT_PATH, '社交媒体_最终分析数据库.xlsx')
                df_final.to_excel(final_path, index=False)
                UX.info(f"最终分析数据库路径: {final_path}")
                
                # 统计
                if 'processing_status' in df_final.columns:
                    success = (df_final['processing_status'] == ProcessingStatus.SUCCESS).sum()
                    failed = df_final['processing_status'].isin([ProcessingStatus.STAGE_1_FAILED, ProcessingStatus.STAGE_2_FAILED]).sum()
                    no_relevant = (df_final['processing_status'] == ProcessingStatus.NO_RELEVANT).sum()
                    UX.ok(f"最终数据库 | 总计 {len(df_final)} | 成功 {success} | 失败 {failed} | 无相关 {no_relevant}")
                else:
                    UX.warn("最终数据库中没有processing_status列")
            else:
                UX.warn("没有成功加载任何结果文件")
        else:
            UX.warn("没有结果文件需要合并")
        
        # 生成信度检验文件
        import traceback
        try:
            result_files = glob.glob(os.path.join(OUTPUT_PATH, '*analyzed_*.xlsx'))
            input_files = glob.glob(os.path.join(INPUT_PATH, '*.xlsx'))

            if not result_files or not input_files:
                UX.warn('缺少信度检验所需的输入/结果文件')
            else:
                result_frames = []
                for path in result_files:
                    df_res = pd.read_excel(path)
                    # 不覆盖Excel里已有的Source，处理器已写入正确值
                    # source_name = os.path.basename(path).replace('(不能删)analyzed_', '')
                    # df_res['Source'] = identify_source(source_name)
                    result_frames.append(df_res)
                df_all_results = pd.concat(result_frames, ignore_index=True)

                input_frames = []
                for path in input_files:
                    df_in = pd.read_excel(path)
                    df_in['Source'] = identify_source(os.path.basename(path))
                    input_frames.append(df_in)
                df_all_input = pd.concat(input_frames, ignore_index=True)

                from core.reliability import create_reliability_test_module

                reliability_module = create_reliability_test_module(
                    output_path=OUTPUT_PATH,
                    sampling_config=RELIABILITY_SAMPLING_CONFIG,
                    random_seed=CONFIG.get('project', {}).get('random_seed', 42)
                )

                reliability_module.generate_files(df_all_results, df_all_input)
        except Exception as e:
            UX.err(f"信度检验文件生成失败: {e}")
            UX.err(traceback.format_exc())
    else:
        UX.info("信度检验模式已关闭，跳过信度检验文件生成")
    
    UX.phase("所有任务完成")

async def main():
    """主函数"""
    UX.start_run()
    
    # 创建自动重试管理器
    retry_manager = create_auto_retry_manager(CONFIG, OUTPUT_PATH)
    
    # 使用自动重试管理器执行主处理逻辑
    await retry_manager.run_with_auto_retry(main_processing_logic)

if __name__ == "__main__":
    import sys
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())