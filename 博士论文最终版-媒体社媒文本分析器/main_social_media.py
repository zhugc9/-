# -*- coding: utf-8 -*-
"""
社交媒体分析主入口 - VK和知乎数据处理
"""
import os
import asyncio
import aiohttp
import pandas as pd
import glob

# 导入核心模块
from core import (
    UX, load_config, APIService, SyncAPIService,
    VKProcessor, ZhihuProcessor, identify_source, count_tokens
)
from core.processors import ProcessingStatus

# 加载配置
CONFIG = load_config()
if CONFIG is None:
    raise RuntimeError("无法加载配置文件，程序退出")

# 从配置中提取变量（兼容社交媒体项目的配置结构）
INPUT_PATH = CONFIG.get('INPUT_PATH', CONFIG.get('file_paths', {}).get('input', ''))
OUTPUT_PATH = CONFIG.get('OUTPUT_PATH', CONFIG.get('file_paths', {}).get('output', ''))
RELIABILITY_TEST_MODE = CONFIG.get('RELIABILITY_TEST_MODE', CONFIG.get('reliability_test', {}).get('enabled', False))
RELIABILITY_SAMPLING_CONFIG = CONFIG.get('RELIABILITY_SAMPLING_CONFIG', CONFIG.get('reliability_test', {}).get('sampling_config', {}))

# 社交媒体特有配置
MAX_CONCURRENT_REQUESTS = CONFIG.get('MAX_CONCURRENT_REQUESTS', 2)
API_RETRY_ATTEMPTS = CONFIG.get('API_RETRY_ATTEMPTS', 3)
RATE_LIMIT_BASE_DELAY = CONFIG.get('RATE_LIMIT_BASE_DELAY', 2)
SKIP_FAILED_TEXTS = CONFIG.get('SKIP_FAILED_TEXTS', True)
VK_LONG_TEXT_THRESHOLD = CONFIG.get('VK_LONG_TEXT_THRESHOLD', 1500)
ZHIHU_SHORT_TOKEN_THRESHOLD = CONFIG.get('ZHIHU_SHORT_TOKEN_THRESHOLD', 100)
ZHIHU_LONG_TOKEN_THRESHOLD = CONFIG.get('ZHIHU_LONG_TOKEN_THRESHOLD', 1300)
LANGUAGE_CONFIGS = CONFIG.get('LANGUAGE_CONFIGS', {})
COLUMN_MAPPING = CONFIG.get('COLUMN_MAPPING', {})

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

async def main():
    """主函数"""
    UX.start_run()
    UX.phase("社交媒体议题单元分析器启动")
    UX.info(f"信度检验模式: {'开启' if RELIABILITY_TEST_MODE else '关闭'}")
    
    # 检查配置
    required_models = ["VK_BATCH", "ZHIHU_CHUNKING", "ZHIHU_ANALYSIS"]
    api_config = CONFIG.get('API_CONFIG', CONFIG.get('api_config', {}))
    missing = [m for m in required_models if m not in api_config.get("STAGE_MODELS", {})]
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
    
    # 创建API服务实例（用于统计）
    api_service_sync = SyncAPIService(CONFIG)
    
    # 处理VK文件
    if vk_files:
        UX.phase("处理VK文件")
        
        for filename in vk_files:
            input_file = os.path.join(INPUT_PATH, filename)
            output_file = os.path.join(OUTPUT_PATH, f"(不能删)analyzed_{filename}")
            
            try:
                df = pd.read_excel(input_file)
                processor = VKProcessor(api_service_sync, CONFIG, prompts)
                processor.process(df, output_file, 'vk')
                
                if processor.Units_collector:
                    all_Units_data.extend(processor.Units_collector)
                all_results_files.append(output_file)
                
            except Exception as e:
                UX.err(f"处理VK文件 {filename} 失败: {e}")
    
    # 处理知乎文件
    if zhihu_files:
        UX.phase("处理知乎文件")
        failed_zhihu_ids = []  # 初始化失败任务列表
        
        async with aiohttp.ClientSession() as session:
            api_service_async = APIService(session, CONFIG)
            
            for filename in zhihu_files:
                input_file = os.path.join(INPUT_PATH, filename)
                output_file = os.path.join(OUTPUT_PATH, f"(不能删)analyzed_{filename}")
                
                try:
                    df = pd.read_excel(input_file)
                    processor = ZhihuProcessor(api_service_async, CONFIG, prompts)
                    # 接收返回的失败ID
                    failed_ids_in_file = await processor.process(df, output_file, '知乎')
                    if failed_ids_in_file:
                        failed_zhihu_ids.extend(failed_ids_in_file)
                    
                    if processor.Units_collector:
                        all_Units_data.extend(processor.Units_collector)
                    all_results_files.append(output_file)
                    
                except Exception as e:
                    UX.err(f"处理知乎文件 {filename} 失败: {e}")
                    # 记录文件级别的失败
                    failed_zhihu_ids.append(f"FILE_FAILED: {filename} - {str(e)[:100]}")
            
            # 打印API统计
            api_service_async.print_statistics()
        
        # 检查并记录知乎失败任务
        if failed_zhihu_ids:
            log_path = os.path.join(OUTPUT_PATH, 'zhihu_failed_ids_log.txt')
            try:
                with open(log_path, 'w', encoding='utf-8') as f:
                    for failed_id in failed_zhihu_ids:
                        f.write(f"{failed_id}\n")
                UX.warn(f"知乎失败任务已记录到: {log_path}")
            except Exception as e:
                UX.warn(f"写入知乎失败日志失败: {e}")
    
    # 打印VK处理的API统计
    api_service_sync.print_statistics()
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
                    UX.info(f"加载结果文件: {file} ({len(df)}条记录)")
                else:
                    UX.warn(f"结果文件不存在: {file}")
            
            if all_results:
                df_final = pd.concat(all_results, ignore_index=True)
                
                # --- BEGIN INSERTED BLOCK ---
                UX.info("正在标准化文本列名...")
                # 检查'Unit_Text'列是否存在，如果不存在则创建，避免后续操作因列不存在而失败
                if 'Unit_Text' not in df_final.columns:
                    df_final['Unit_Text'] = pd.NA

                # 定位到所有Source为'vk'的行
                vk_mask = df_final['Source'] == 'vk'

                # 将这些行中'comment_text'列的内容，填充到'Unit_Text'列的空值位置
                # 使用.loc确保安全赋值
                df_final.loc[vk_mask, 'Unit_Text'] = df_final.loc[vk_mask, 'Unit_Text'].fillna(df_final.loc[vk_mask, 'comment_text'])

                UX.ok("文本列名'Unit_Text'标准化完成。")
                # --- END INSERTED BLOCK ---
                
                final_path = os.path.join(OUTPUT_PATH, '社交媒体_最终分析数据库.xlsx')
                df_final.to_excel(final_path, index=False)
                UX.info(f"最终分析数据库路径: {final_path}")
                
                # 统计
                if 'processing_status' in df_final.columns:
                    success = (df_final['processing_status'] == ProcessingStatus.SUCCESS).sum()
                    failed = (df_final['processing_status'] == ProcessingStatus.API_FAILED).sum()
                    no_relevant = (df_final['processing_status'] == ProcessingStatus.NO_RELEVANT).sum()
                    UX.ok(f"最终数据库: 总{len(df_final)}条, 成功{success}, 失败{failed}, 无相关内容{no_relevant}")
                else:
                    UX.warn("最终数据库中没有processing_status列")
            else:
                UX.warn("没有成功加载任何结果文件")
        else:
            UX.warn("没有结果文件需要合并")
        
        # 生成信度检验文件
        if final_path:
            try:
                UX.info("开始生成信度检验文件...")
                from core.reliability import create_reliability_test_module
                
                reliability_module = create_reliability_test_module(
                    input_path=INPUT_PATH,
                    output_path=OUTPUT_PATH,
                    sampling_config=RELIABILITY_SAMPLING_CONFIG,
                    id_column=COLUMN_MAPPING.get('ID', 'ID'),
                    text_column=COLUMN_MAPPING.get('MEDIA_TEXT', 'Unit_Text')
                )
                
                reliability_module.generate_reliability_files()
                UX.ok("信度检验文件生成完成")
            except Exception as e:
                UX.err(f"生成信度检验文件失败: {str(e)}")
                import traceback
                UX.err(f"详细错误信息: {traceback.format_exc()}")
        else:
            UX.warn(f"无法生成信度检验文件 - 最终结果文件: {final_path}")
    else:
        UX.info("信度检验模式已关闭，跳过信度检验文件生成")
    
    # 配置验证测试
    UX.info("=== 配置验证测试 ===")
    UX.info(f"VK长文本阈值: {VK_LONG_TEXT_THRESHOLD} tokens")
    UX.info(f"知乎短文本阈值: {ZHIHU_SHORT_TOKEN_THRESHOLD} tokens")  
    UX.info(f"知乎长文本阈值: {ZHIHU_LONG_TOKEN_THRESHOLD} tokens")
    UX.info(f"最大并发请求数: {MAX_CONCURRENT_REQUESTS}")
    UX.info(f"API重试次数: {API_RETRY_ATTEMPTS}")
    
    # 测试token计算器
    test_text = "这是一个测试文本用来验证token计算功能是否正常工作"
    test_tokens = count_tokens(test_text)
    UX.info(f"测试文本token数: {test_tokens}")
    
    UX.phase("所有任务完成")

if __name__ == "__main__":
    import sys
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())