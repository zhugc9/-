# -*- coding: utf-8 -*-
"""
==============================================================================
长文本分析器 - 新闻专用引擎 (种子扩展一体化+议题单元）
==============================================================================
"""
import os
import re
import glob
import json
import time
import asyncio
import aiohttp
import pandas as pd
import hashlib
import yaml
import tiktoken
from datetime import datetime
from tqdm.asyncio import tqdm as aio_tqdm
from asyncio import Lock
import contextlib

class UX:
    @staticmethod
    def _fmt(msg):
        return str(msg).strip()
    
    # 运行级计时起点
    RUN_T0 = time.perf_counter()
    
    @staticmethod
    def start_run():
        UX.RUN_T0 = time.perf_counter()

    @staticmethod
    def _ts():
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    @staticmethod
    def _elapsed():
        return time.perf_counter() - UX.RUN_T0
    
    @staticmethod
    def _elapsed_str():
        total = int(UX._elapsed())
        days, rem = divmod(total, 86400)
        hours, rem = divmod(rem, 3600)
        minutes, seconds = divmod(rem, 60)
        if days > 0:
            return f"{days}天{hours:02d}:{minutes:02d}:{seconds:02d}"
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    @staticmethod
    def phase(title):
        print(f"\n=== [{UX._ts()}][{UX._elapsed_str()}] {UX._fmt(title)} ===")
    
    @staticmethod
    def info(msg):
        print(f"[{UX._ts()}][{UX._elapsed_str()}] [i] {UX._fmt(msg)}")
    
    @staticmethod
    def ok(msg):
        print(f"[{UX._ts()}][{UX._elapsed_str()}] [OK] {UX._fmt(msg)}")
    
    @staticmethod
    def warn(msg):
        print(f"[{UX._ts()}][{UX._elapsed_str()}] [!] {UX._fmt(msg)}")
    
    @staticmethod
    def err(msg):
        print(f"[{UX._ts()}][{UX._elapsed_str()}] [!!!] {UX._fmt(msg)}")
    
    @staticmethod
    @contextlib.contextmanager
    def timer(label):
        t0 = time.perf_counter()
        UX.info(f"{label} 开始")
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            total = int(dt)
            h, rem = divmod(total, 3600)
            m, s = divmod(rem, 60)
            UX.ok(f"{label} 完成，用时 {h:02d}:{m:02d}:{s:02d}")

# ==============================================================================
# === 🎛️ 配置加载区 ================================
# ==============================================================================
def load_config():
    """加载配置文件"""
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 环境变量兜底覆盖
        env_api_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
        if env_api_key:
            try:
                if isinstance(config.get("api_config", {}).get("API_KEYS"), list) and config["api_config"]["API_KEYS"]:
                    config["api_config"]["API_KEYS"][0] = env_api_key
                else:
                    config["api_config"]["API_KEYS"] = [env_api_key]
            except Exception:
                pass
        
        env_base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("API_BASE_URL")
        if env_base_url:
            try:
                config["api_config"]["BASE_URL"] = env_base_url.rstrip("/")
            except Exception:
                pass
        
        return config
    except Exception as e:
        print(f"加载配置文件失败: {e}")
        return None

# 加载配置
CONFIG = load_config()
if CONFIG is None:
    raise RuntimeError("无法加载配置文件，程序退出")

# 从配置中提取变量（保持向后兼容）
INPUT_PATH = CONFIG['file_paths']['input']
OUTPUT_PATH = CONFIG['file_paths']['output']
RELIABILITY_TEST_MODE = CONFIG['reliability_test']['enabled']
RELIABILITY_SAMPLING_CONFIG = CONFIG['reliability_test']['sampling_config']
LANGUAGE_CONFIGS = CONFIG['LANGUAGE_CONFIGS']
SKIP_FAILED_TEXTS = CONFIG['data_processing']['skip_failed_texts']
API_CONFIG = CONFIG['api_config']
COLUMN_MAPPING = CONFIG['column_mapping']
REQUIRED_OUTPUT_COLUMNS = CONFIG['required_output_columns']

# 从简化的配置结构中提取
API_RETRY_CONFIG = CONFIG.get('api_retry_config', {})
MODEL_POOLS = CONFIG.get('model_pools', {})

# 具体值
MAX_CONCURRENT_REQUESTS = API_RETRY_CONFIG.get('max_concurrent_requests', 1)
API_RETRY_ATTEMPTS = API_RETRY_CONFIG.get('attempts_per_model', 3)
RETRY_DELAYS = API_RETRY_CONFIG.get('retry_delays', [2, 5, 10])
MAX_MODEL_SWITCHES = API_RETRY_CONFIG.get('max_model_switches', 10)
QUEUE_TIMEOUT = API_RETRY_CONFIG.get('queue_timeout', 30.0)

# 其他配置项
BUFFER_CONFIG = CONFIG.get('buffer_config', {})
API_REQUEST_PARAMS = CONFIG.get('api_request_params', {})
QUALITY_THRESHOLDS = CONFIG.get('data_processing', {}).get('quality_thresholds', {})
RANDOMIZATION_CONFIG = CONFIG.get('randomization', {})
SHORT_TEXT_THRESHOLD = CONFIG.get('data_processing', {}).get('SHORT_TEXT_THRESHOLD', 100)
LONG_TEXT_THRESHOLD = CONFIG.get('data_processing', {}).get('LONG_TEXT_THRESHOLD', 1200)

# 批处理配置
ENABLE_BATCH_PROCESSING = CONFIG.get('data_processing', {}).get('enable_batch_processing', False)
BATCH_SIZE = CONFIG.get('data_processing', {}).get('batch_size', 5)

def ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    """确保结果表具备统一列：缺失则以空值补齐，不移除已有列。该操作是无破坏性的：仅添加缺列，避免合并时列被丢失。"""
    if df is None or df.empty:
        # 即使空表，也返回包含所有列的空DataFrame，保证下游concat有列头
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

# === 统一状态定义（与社交媒体代码保持一致）===
class ProcessingStatus:
    """统一的处理状态定义"""
    SUCCESS = "SUCCESS"              # 成功处理
    NO_RELEVANT = "NO_RELEVANT"      # 无相关内容（不需要重试）
    API_FAILED = "API_FAILED"        # API调用失败（需要重试）

# ==============================================================================
# === 🔧 核心功能模块 =========================================================
# ==============================================================================

# Token计算器初始化
_tokenizer = None

def get_tokenizer():
    """获取token计算器，使用单例模式"""
    global _tokenizer
    if _tokenizer is None:
        try:
            _tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            UX.err(f"初始化tokenizer失败: {e}")
            raise
    return _tokenizer

def count_tokens(text: str) -> int:
    """计算文本的token数量"""
    if not isinstance(text, str) or not text.strip():
        return 0
    
    try:
        tokenizer = get_tokenizer()
        return len(tokenizer.encode(text))
    except Exception as e:
        UX.warn(f"计算token数失败: {e}")
        # 降级策略：使用字符数的粗略估算
        token_char_ratio = QUALITY_THRESHOLDS.get('token_char_ratio', 2)
        return len(text) // token_char_ratio  # 从配置读取token与字符比例

def get_processing_state(df, id_col):
    """统一的状态检查：返回(完全成功ID集合, 有失败的ID集合)"""
    if df is None or df.empty or id_col not in df.columns:
        return set(), set()
    
    status_col = 'processing_status'
    try:
        if status_col in df.columns:
            # 🔧 修复：基于文章维度判断状态
            fully_successful_ids = set()
            has_failed_ids = set()
            
            # 按文章ID分组统计状态
            for article_id in df[id_col].unique():
                article_records = df[df[id_col] == article_id]
                statuses = article_records[status_col].tolist()
                
                # 检查是否有失败记录
                if ProcessingStatus.API_FAILED in statuses:
                    has_failed_ids.add(str(article_id))
                # 检查是否所有记录都是成功或无相关
                elif all(s in [ProcessingStatus.SUCCESS, ProcessingStatus.NO_RELEVANT] for s in statuses):
                    fully_successful_ids.add(str(article_id))
                # 其他情况（比如只有部分记录）暂不分类
                
            return fully_successful_ids, has_failed_ids
        else:
            # 兼容旧版
            speaker_col = 'speaker'
            if speaker_col in df.columns:
                success = df[~df[speaker_col].astype(str).str.contains('API_CALL_FAILED', na=False)][id_col]
                failed = df[df[speaker_col].astype(str).str.contains('API_CALL_FAILED', na=False)][id_col]
                return set(success.astype(str)), set(failed.astype(str))
            else:
                return set(), set()
        
    except Exception:
        return set(), set()

def get_stage_model(stage_key: str) -> str:
    try:
        # 优先使用新的模型池配置
        if MODEL_POOLS and 'primary_models' in MODEL_POOLS:
            return MODEL_POOLS['primary_models'][stage_key]
        # 向后兼容：使用旧的配置
        return API_CONFIG["STAGE_MODELS"][stage_key]
    except Exception:
        raise ValueError(f"未配置阶段模型: {stage_key}")

file_write_lock = Lock()

def safe_str_convert(value):
    if value is None:
        return ''
    if isinstance(value, pd.Series):
        if value.empty:
            return ''
        return str(value.iloc[0])
    if pd.isna(value):
        return ''
    return str(value)

def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        text = safe_str_convert(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def detect_language_and_get_config(text):
    if re.search(r'[\u4e00-\u9fa5]', text):
        return 'zh', LANGUAGE_CONFIGS['zh']
    elif re.search(r'[\u0400-\u04FF]', text):
        return 'ru', LANGUAGE_CONFIGS['ru']
    else:
        return 'en', LANGUAGE_CONFIGS['en']

def safe_get_speaker(data, fallback="未知信源"):
    try:
        if not isinstance(data, dict):
            return fallback
        
        speaker_raw = data.get('speaker')
        if speaker_raw is None:
            return fallback
        
        if not isinstance(speaker_raw, str):
            speaker_raw = str(speaker_raw)
        
        speaker_clean = speaker_raw.strip()
        # 移除常见的无效值
        if not speaker_clean or speaker_clean in ["speaker", '"speaker"', "'speaker'"]:
            return fallback

        # 去除引号
        if len(speaker_clean) >= 2:
            if (speaker_clean[0] == speaker_clean[-1]) and speaker_clean[0] in ['"', "'"]:
                speaker_clean = speaker_clean[1:-1].strip()

        return speaker_clean if speaker_clean else fallback
    except Exception:
        return fallback

def create_unified_record(record_type, original_id, source="未知来源", text_snippet="", failure_reason=""):
    base_record = {
        "processing_status": record_type,
        "Source": source,
        "Unit_ID": f"{original_id}-{record_type}"
    }

    if record_type == ProcessingStatus.NO_RELEVANT:
        return {
            **base_record,
            "Macro_Chunk_ID": "NO_RELEVANT",
            "speaker": "NO_RELEVANT_CONTENT",
            "Unit_Text": "[无相关内容]",
            "Incident": "",
            "Frame_ProblemDefinition": "",
            "Frame_ResponsibilityAttribution": "",
            "Frame_MoralEvaluation": "",
            "Frame_SolutionRecommendation": "",
            "Frame_ActionStatement": "",
            "Frame_CausalExplanation": "",
            "Valence": "",
            "Evidence_Type": "",
            "Attribution_Level": "",
            "Temporal_Focus": "",
            "Primary_Actor_Type": "",
            "Geographic_Scope": "",
            "Relationship_Model_Definition": "",
            "Discourse_Type": ""
        }

    elif record_type == ProcessingStatus.API_FAILED:
        return {
            **base_record,
            "Macro_Chunk_ID": "API_FAILED",
            "speaker": "API_CALL_FAILED",
            "Unit_Text": f"[API_FAILED] {failure_reason}: {text_snippet[:200]}...",
            "Incident": "API_CALL_FAILED",
            "Frame_ProblemDefinition": "[]",
            "Frame_ResponsibilityAttribution": "[]",
            "Frame_MoralEvaluation": "[]",
            "Frame_SolutionRecommendation": "[]",
            "Frame_ActionStatement": "[]",
            "Frame_CausalExplanation": "[]",
            "Valence": "API_CALL_FAILED",
            "Evidence_Type": "API_CALL_FAILED",
            "Attribution_Level": "API_CALL_FAILED",
            "Temporal_Focus": "API_CALL_FAILED",
            "Primary_Actor_Type": "API_CALL_FAILED",
            "Geographic_Scope": "API_CALL_FAILED",
            "Relationship_Model_Definition": "API_CALL_FAILED",
            "Discourse_Type": "API_CALL_FAILED"
        }

def clean_failed_records(output_path, id_column):
    if not os.path.exists(output_path):
        return set()

    try:
        df = pd.read_excel(output_path)
        if df.empty or id_column not in df.columns:
            return set()

        if 'processing_status' in df.columns:
            # 修正：只清理API_FAILED的记录，保留NO_RELEVANT记录
            failed_mask = df['processing_status'] == ProcessingStatus.API_FAILED
        else:
            failed_mask = df['speaker'].astype(str).str.contains('API_CALL_FAILED', na=False) if 'speaker' in df.columns else pd.Series([False] * len(df))

        failed_ids = set(df[failed_mask][id_column].astype(str).unique())
        if failed_ids:
            df_clean = df[~failed_mask]
            df_clean.to_excel(output_path, index=False)

            if 'processing_status' in df_clean.columns:
                success_count = (df_clean['processing_status'] == ProcessingStatus.SUCCESS).sum()
                no_relevant_count = (df_clean['processing_status'] == ProcessingStatus.NO_RELEVANT).sum()
                UX.info(f"保留了 {success_count} 条成功记录, {no_relevant_count} 条无相关记录")

            UX.info(f"清理了 {len(failed_ids)} 个API失败记录，剩余 {len(df_clean)} 条")

        return failed_ids

    except Exception as e:
        UX.warn(f"清理失败记录时出错: {e}")
        return set()

def identify_source(filename):
    source_map = {
        '俄总统': ['俄总统', '总统', 'Putin', 'president'],
        '俄语媒体': ['俄语媒体', '俄语', 'russian', 'ru_media', '俄媒'],
        '中文媒体': ['中文媒体', '中文', 'chinese', 'cn_media', '中媒'],
        '英语媒体': ['英语媒体', '英语', 'english', 'en_media', '英媒']
    }

    filename_lower = filename.lower()
    for source, keywords in source_map.items():
        if any(kw.lower() in filename_lower for kw in keywords):
            UX.info(f"文件 {filename} 识别为: {source}")
            return source

    UX.warn(f"文件 {filename} 无法识别信源，标记为: 未知来源")
    return '未知来源'

def save_macro_chunks_database(new_macro_chunks_list, database_path):
    """分批次、增量式地保存宏观块到主数据库"""
    if not new_macro_chunks_list:
        return
    try:
        df_new = pd.DataFrame(new_macro_chunks_list)
        if df_new.empty:
            return
        if os.path.exists(database_path):
            try:
                # 读取现有数据并追加，然后去重
                df_existing = pd.read_excel(database_path)
                df_final = pd.concat([df_existing, df_new], ignore_index=True)
                df_final = df_final.drop_duplicates(subset=['Macro_Chunk_ID'], keep='last')
            except Exception as e:
                UX.warn(f"读取现有宏观块数据库失败: {e}，将覆盖写入")
                df_final = df_new
        else:
            df_final = df_new
        df_final.to_excel(database_path, index=False)
        UX.ok(f"宏观块数据库已更新: {database_path} (累计 {len(df_final)} 条)")
    except Exception as e:
        UX.err(f"保存宏观块数据库失败: {e}")

def get_existing_macro_chunks(original_id, macro_db_path):
    """获取已存在的宏观块信息（包含永久ID），用于重新分析时保持一致性"""
    if not os.path.exists(macro_db_path):
        return None
    try:
        df_macro = pd.read_excel(macro_db_path)
        if df_macro.empty or 'Original_ID' not in df_macro.columns:
            return None
        # 查找该原始ID对应的宏观块
        existing_chunks_df = df_macro[df_macro['Original_ID'].astype(str) == str(original_id)]
        if existing_chunks_df.empty:
            return None
        
        # 将DataFrame直接转换为字典列表，保留所有列（特别是Macro_Chunk_ID）
        macro_chunks = existing_chunks_df.to_dict('records')
        UX.info(f"找到ID {original_id} 的已保存宏观块: {len(macro_chunks)} 个")
        return macro_chunks
    except Exception as e:
        UX.warn(f"读取已保存宏观块失败: {e}")
        return None

def build_resume_plan(output_file_path: str, df_input: pd.DataFrame, id_col: str):
    """构建断点续传计划
    返回:
    - never_processed_ids: 从未处理过的文章ID集合
    - rechunk_article_ids: 需要重新切分宏观块的文章ID集合（全有或全无）
    - macro_chunks_to_rerun: dict[original_id] -> set(Macro_Chunk_ID) 需重新分析的宏观块集合
    """
    all_input_ids = set(df_input[id_col].astype(str)) if (df_input is not None and id_col in df_input.columns) else set()

    never_processed_ids = set(all_input_ids)
    rechunk_article_ids = set()
    macro_chunks_to_rerun = {}

    if os.path.exists(output_file_path):
        try:
            df_existing = pd.read_excel(output_file_path)
            if not df_existing.empty and id_col in df_existing.columns:
                processed_article_ids = set(df_existing[id_col].astype(str).unique())
                never_processed_ids = all_input_ids - processed_article_ids

                if 'processing_status' in df_existing.columns:
                    failed_records = df_existing[df_existing['processing_status'] == ProcessingStatus.API_FAILED]
                    for _, record in failed_records.iterrows():
                        macro_chunk_id = str(record.get('Macro_Chunk_ID', '')).strip()
                        article_id = str(record.get(id_col, '')).strip()

                        # 兼容：旧数据未填特殊标记，仅记录了失败文案
                        if not macro_chunk_id or macro_chunk_id == 'API_FAILED':
                            unit_text = str(record.get('Unit_Text', '')).strip()
                            # 优先：从文案中解析形如 ABC123-M45 的宏观块编号
                            # 避免误匹配 MACRO_CHUNKING_FAILED
                            try:
                                match = re.search(r'([A-Za-z0-9_-]+-M\d+)', unit_text)
                            except Exception:
                                match = None
                            if match:
                                macro_chunk_id = match.group(1)
                            else:
                                # 若能识别到"切分失败"字样，则按文章级重切分处理
                                if ('切分' in unit_text and '失败' in unit_text) or ('宏观块切分失败' in unit_text):
                                    if article_id:
                                        rechunk_article_ids.add(article_id)
                                    continue
                                # 无法定位宏观块编号且不是明确的切分失败 -> 跳过该条，等待下一轮
                                continue

                        if macro_chunk_id.endswith('-MACRO_CHUNKING_FAILED'):
                            if article_id:
                                rechunk_article_ids.add(article_id)
                        elif macro_chunk_id and macro_chunk_id != 'API_FAILED':
                            s = macro_chunks_to_rerun.setdefault(article_id, set())
                            s.add(macro_chunk_id)
        except Exception as e:
            UX.warn(f"读取已处理文件失败: {e}")

    return never_processed_ids, rechunk_article_ids, macro_chunks_to_rerun

def get_failed_macro_chunk_ids(original_id, output_file_path):

    """获取该文章中失败的宏观块ID列表"""
    if not os.path.exists(output_file_path):
        return set()
    try:
        df_output = pd.read_excel(output_file_path)
        if df_output.empty:
            return set()
        # 查找该原始ID对应的失败记录
        id_mask = df_output[COLUMN_MAPPING["ID"]].astype(str) == str(original_id)
        failed_mask = df_output['processing_status'] == ProcessingStatus.API_FAILED
        failed_records = df_output[id_mask & failed_mask]
        if failed_records.empty:
            return set()
        # 提取失败的宏观块ID
        failed_macro_chunk_ids = set()
        if 'Macro_Chunk_ID' in failed_records.columns:
            failed_macro_chunk_ids = set(failed_records['Macro_Chunk_ID'].dropna().astype(str))
        return failed_macro_chunk_ids
    except Exception as e:
        UX.warn(f"读取失败宏观块ID失败: {e}")
        return set()

def get_successful_macro_chunk_ids(original_id, output_file_path):
    """获取该文章中已成功处理的宏观块ID列表"""
    if not os.path.exists(output_file_path):
        return set()
    try:
        df_output = pd.read_excel(output_file_path)
        if df_output.empty:
            return set()
        # 查找该原始ID对应的成功记录
        id_mask = df_output[COLUMN_MAPPING["ID"]].astype(str) == str(original_id)
        success_mask = df_output['processing_status'] == ProcessingStatus.SUCCESS
        success_records = df_output[id_mask & success_mask]
        if success_records.empty:
            return set()
        # 提取成功的宏观块ID
        successful_macro_chunk_ids = set()
        if 'Macro_Chunk_ID' in success_records.columns:
            successful_macro_chunk_ids = set(success_records['Macro_Chunk_ID'].dropna().astype(str))
        return successful_macro_chunk_ids
    except Exception as e:
        UX.warn(f"读取成功宏观块ID失败: {e}")
        return set()

def needs_reprocessing(original_id, output_file_path, macro_db_path):
    """判断文章是否需要重新处理（支持宏观块级别的精确判断）"""
    # 如果没有分析结果文件，需要处理
    if not os.path.exists(output_file_path):
        return True
    
    # 如果没有宏观块数据库，按原有逻辑处理
    if not macro_db_path or not os.path.exists(macro_db_path):
        return True
    
    try:
        # 获取该文章的所有宏观块
        existing_macro_chunks = get_existing_macro_chunks(original_id, macro_db_path)
        if not existing_macro_chunks:
            return True
        
        # 获取所有应该存在的宏观块ID
        expected_macro_chunk_ids = set()
        for i in range(len(existing_macro_chunks)):
            expected_macro_chunk_ids.add(f"{original_id}-M{i+1}")
        
        # 获取已成功处理的宏观块ID
        successful_macro_chunk_ids = get_successful_macro_chunk_ids(original_id, output_file_path)
        
        # 获取失败的宏观块ID  
        failed_macro_chunk_ids = get_failed_macro_chunk_ids(original_id, output_file_path)
        
        # 🔧 修复逻辑：更精确的判断
        # 1. 检查是否有宏观块切分失败（需要重新处理整篇文章）
        macro_chunking_failed = any(chunk_id.endswith('-MACRO_CHUNKING_FAILED') for chunk_id in failed_macro_chunk_ids)
        if macro_chunking_failed:
            UX.info(f"文章 {original_id}: 检测到宏观块切分失败，需要重新处理整篇文章")
            return True
        
        # 2. 检查是否所有宏观块都已成功处理
        missing_macro_chunks = expected_macro_chunk_ids - successful_macro_chunk_ids
        has_missing = len(missing_macro_chunks) > 0
        has_failed = len(failed_macro_chunk_ids) > 0
        
        if has_failed or has_missing:
            UX.info(f"文章 {original_id}: 需要重新处理部分宏观块 (失败:{len(failed_macro_chunk_ids)}个, 缺失:{len(missing_macro_chunks)}个)")
            return True
        
        # 3. 所有宏观块都已成功处理
        UX.info(f"文章 {original_id}: 所有{len(expected_macro_chunk_ids)}个宏观块均已成功处理，跳过")
        return False
        
    except Exception as e:
        UX.warn(f"判断文章{original_id}是否需要重新处理时出错: {e}")
        return True
        
class APIService:
    def __init__(self, session):
        self.session = session
        self.fail_counts = {}
        self.model_index = {}
        self.total_switches = {}

    def _select_model(self, stage_key: str, explicit_model: str = None) -> str:
        if explicit_model:
            return explicit_model
        
        # 获取主模型和备用模型
        if MODEL_POOLS and 'primary_models' in MODEL_POOLS:
            primary = MODEL_POOLS['primary_models'].get(stage_key)
            fallback_list = MODEL_POOLS.get('fallback_models', {}).get(stage_key, [])
            fallback = fallback_list[0] if fallback_list else primary  # 没有备用就用主模型
        else:
            # 向后兼容：使用旧的配置
            primary = get_stage_model(stage_key)
            fallback_list = API_CONFIG.get('FALLBACK', {}).get('STAGE_CANDIDATES', {}).get(stage_key, [])
            fallback = fallback_list[0] if fallback_list else primary
        
        # 如果主备相同，就不切换
        if primary == fallback:
            return primary
        
        models = [primary, fallback]
        idx = self.model_index.get(stage_key, 0)
        return models[idx]

    def _handle_failure(self, stage_key: str) -> bool:
        """3次失败后切换模型"""
        # 切换计数
        self.total_switches[stage_key] = self.total_switches.get(stage_key, 0) + 1
        
        # 检查是否达到上限（20次切换 = 10轮）
        if self.total_switches[stage_key] >= MAX_MODEL_SWITCHES:
            return False
        
        # 切换模型：主模型 ←→ 备用模型
        self.model_index[stage_key] = 1 - self.model_index.get(stage_key, 0)
        UX.info(f"[{stage_key}] 切换模型 (第{self.total_switches[stage_key]}次)")
        return True

    async def _call_api(self, prompt, language='zh', model_name=None, stage_key=None, context_label=None):
        # 初始化变量，避免在异常处理中引用未定义的变量
        response = None
        url = ""
        config = LANGUAGE_CONFIGS.get(language, LANGUAGE_CONFIGS['zh'])
        prompt_tokens = count_tokens(prompt)
        
        # 从统一超时配置读取token阈值和超时时间（所有语言通用）
        timeout_config = CONFIG['TIMEOUT_CONFIG']
        threshold_short = timeout_config['TOKEN_THRESHOLD_SHORT']
        threshold_medium = timeout_config['TOKEN_THRESHOLD_MEDIUM']
        
        if prompt_tokens < threshold_short:
            timeout = timeout_config['TIMEOUT_SHORT']  # 短文本
        elif prompt_tokens <= threshold_medium:
            timeout = timeout_config['TIMEOUT_MEDIUM']  # 中等文本
        else:
            timeout = timeout_config['TIMEOUT_LONG']  # 长文本
        
        for attempt in range(API_RETRY_ATTEMPTS):
            try:
                api_key = API_CONFIG["API_KEYS"][0]
                url = f"{API_CONFIG['BASE_URL']}/chat/completions"
                headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
                chosen_model = self._select_model(stage_key or 'INTEGRATED_ANALYSIS', model_name)
                
                # 从配置读取API请求参数
                payload = {
                    "model": chosen_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": API_REQUEST_PARAMS.get('temperature', 0),
                    "response_format": API_REQUEST_PARAMS.get('response_format', {"type": "json_object"})
                }
                
                async with self.session.post(url, headers=headers, json=payload,
                                           timeout=aiohttp.ClientTimeout(total=timeout)) as response:
                    response_text = await response.text()
                    response.raise_for_status()
                    response_json = json.loads(response_text)
                    content = response_json.get('choices', [{}])[0].get('message', {}).get('content')
                    
                    if not content:
                        raise ValueError("响应缺少content")
                    
                    # 提取JSON
                    first_brace = content.find('{')
                    last_brace = content.rfind('}')
                    
                    if first_brace >= 0 and last_brace > first_brace:
                        cleaned_text = content[first_brace:last_brace+1]
                        result = json.loads(cleaned_text)
                        
                        # 成功重置失败计数
                        if stage_key:
                            self.fail_counts[stage_key] = 0
                        return result
                    
                    raise ValueError("未找到有效JSON结构")
            
            except Exception as e:
                # 记录API失败信息
                status_code = response.status if response else 'N/A'
                
                # 打印原始响应体，便于定位问题
                try:
                    if 'response_text' in locals() and response_text:
                        UX.err(f"API响应体原文: {response_text}")
                    elif response is not None:
                        try:
                            raw_text = await response.text()
                        except Exception:
                            raw_text = "<无法读取响应体>"
                        UX.err(f"API响应体原文: {raw_text}")
                except Exception:
                    pass

                UX.warn(f"[{stage_key or 'API'}] API失败: {status_code}, message='{str(e)}', url='{url}'")
                
                # 重试延迟（除了最后一次）
                if attempt < API_RETRY_ATTEMPTS - 1:
                    delay = RETRY_DELAYS[attempt] if attempt < len(RETRY_DELAYS) else 2
                    await asyncio.sleep(delay)
                # 继续下一次重试
        
        # 3次都失败了，现在处理模型切换
        if stage_key:
            should_continue = self._handle_failure(stage_key)
            if should_continue:
                # 递归调用，用新模型重新试3次
                UX.info(f"[{stage_key}] 使用新模型重新尝试")
                return await self._call_api(prompt, language, None, stage_key, context_label)
            else:
                UX.warn(f"[{stage_key}] 达到切换上限，放弃")
        
        return None

    async def get_analysis(self, prompt, expected_key, language='zh', model_name=None, stage_key=None, context_label=None):
        result = await self._call_api(prompt, language, model_name, stage_key, context_label)
        
        if result is None:
            return None
        
        
        if not isinstance(result, dict) or expected_key not in result:
            if SKIP_FAILED_TEXTS:
                return None
            raise ValueError(f"返回JSON缺少键: '{expected_key}'")
        
        return result[expected_key]

# ==============================================================================
# === 🤖 AI指令模板 =================================================
# ==============================================================================
class Prompts:
    def __init__(self):
        """从文件加载提示词"""
        self.prompts_dir = os.path.join(os.path.dirname(__file__), 'prompts')
        self._load_prompts()

    def _load_prompts(self):
        """加载所有提示词文件"""
        prompt_files = {
            'MACRO_CHUNKING_ZH': 'macro_chunking_zh.txt',
            'MACRO_CHUNKING_EN': 'macro_chunking_en.txt',
            'MACRO_CHUNKING_RU': 'macro_chunking_ru.txt',
            'INTEGRATED_ANALYSIS_V2': 'integrated_analysis_v2.txt'
        }

        for attr_name, filename in prompt_files.items():
            file_path = os.path.join(self.prompts_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    setattr(self, f'_{attr_name}', f.read())
            except Exception as e:
                print(f"加载提示词文件 {filename} 失败: {e}")
                setattr(self, f'_{attr_name}', "")

    # 为了向后兼容，保留属性访问方式
    @property
    def MACRO_CHUNKING_ZH(self):
        return getattr(self, '_MACRO_CHUNKING_ZH', "")

    @property
    def MACRO_CHUNKING_EN(self):
        return getattr(self, '_MACRO_CHUNKING_EN', "")

    @property
    def MACRO_CHUNKING_RU(self):
        return getattr(self, '_MACRO_CHUNKING_RU', "")

    @property
    def INTEGRATED_ANALYSIS_V2(self):
        return getattr(self, '_INTEGRATED_ANALYSIS_V2', "")

# 创建Prompts实例
prompts = Prompts()

# ==============================================================================
# === 📦 批处理专用Prompt模板 ==========================================
# ==============================================================================

PROMPT_BATCH_ANALYSIS = """
# 角色
你是一个高效的、支持并行处理的文本分析引擎。

# 任务
你将收到一个JSON格式的输入，其中包含一个【待分析文本块列表】。
你的任务是，严格按照列表中的顺序，【独立地】对列表中的【每一个】文本块，执行完整的"种子扩展与多维度内容分析"。
你必须为列表中的每一个文本块，都生成一个对应的、完整的分析结果对象。

# 核心规则
- **绝对独立性**: 文本块的分析【绝对不能】受到其他文本块的任何影响。你要像处理N个**完全**独立的API请求一样处理这个列表。它们被打包在一起只是为了提高效率。
- **顺序一致性**: 你的输出结果列表，必须与输入列表的顺序【严格保持一致】。输出列表的第一个元素，必须是输入列表第一个文本块的分析结果。
- **完整性**: 即使某个文本块你认为与中俄无关，或者无法提取出"议题单元"，你也必须为它生成一个对应的空结果对象（例如 `{"analyzed_Units": []}`），以保持位置对应。绝不能因为某个文本块没有结果就跳过它。

# 输入格式示例
```json
{
  "texts_to_analyze": [
    {
      "chunk_id": 0,
      "speaker": "普京 (俄罗斯总统)",
      "text": "这是一个需要分析的短文本块。"
    },
    {
      "chunk_id": 1,
      "speaker": "记者",
      "text": "这是第二个完全独立的短文本块。"
    }
  ]
}
```

# 输出格式
你的回复【必须且只能】是一个JSON对象，包含一个键 "batch_results"。
其值为一个JSON对象的列表，列表中的每一个对象都对应输入列表中的一个文本块。
每一个对象必须包含以下两个键：
- "chunk_id": (整数) 对应输入的chunk_id，用于核对。
- "analysis_output": (JSON对象) 这是对该文本块执行"种子扩展与多维度内容分析"后产出的【完整结果】，其结构必须与你独立分析时输出的 `{"analyzed_Units": [...]}` 完全一致。

# 【注意】：请严格按照上述要求处理下方提供的实际输入。

"""
# ==============================================================================
# === ⚙️ 主处理流程 ==================================================
# ==============================================================================

async def process_chunks_batch(chunks_to_process, original_id, source, api_service):
    """批处理宏观块分析"""
    final_data = []
    
    # 将宏观块分组为批次
    for i in range(0, len(chunks_to_process), BATCH_SIZE):
        batch = chunks_to_process[i:i + BATCH_SIZE]
        UX.info(f"ID {original_id}: 处理批次 {i//BATCH_SIZE + 1}, 包含 {len(batch)} 个宏观块")
        
        # 构建批处理输入
        texts_to_analyze = []
        chunk_mapping = {}  # 保存chunk_id到原始数据的映射
        
        for idx, chunk_data in enumerate(batch):
            chunk_text = chunk_data.get('Macro_Chunk_Text', '').strip()
            if not chunk_text:
                continue
                
            chunk_id = i + idx
            texts_to_analyze.append({
                "chunk_id": chunk_id,
                "speaker": chunk_data.get('Speaker', ''),
                "text": chunk_text
            })
            chunk_mapping[chunk_id] = chunk_data
        
        if not texts_to_analyze:
            continue
            
        # 准备批处理请求
        batch_input = {"texts_to_analyze": texts_to_analyze}
        batch_prompt = PROMPT_BATCH_ANALYSIS + f"\n\n{json.dumps(batch_input, ensure_ascii=False, indent=2)}"
        
        # 检测语言（使用第一个文本块的语言）
        first_text = texts_to_analyze[0]["text"]
        language, _ = detect_language_and_get_config(first_text)
        
        # 批处理专用于超短块，直接使用短文本模型
        model_key = 'INTEGRATED_ANALYSIS_SHORT'
        
        # 执行批处理API调用
        batch_results = await api_service.get_analysis(
            batch_prompt, 'batch_results', language,
            model_name=get_stage_model(model_key), stage_key=f"BATCH_{model_key}",
            context_label=f"{original_id}:BATCH_{i//BATCH_SIZE + 1}"
        )
        
        # 处理批处理结果
        if batch_results and isinstance(batch_results, list):
            for result in batch_results:
                if not isinstance(result, dict):
                    continue
                    
                chunk_id = result.get('chunk_id')
                analysis_output = result.get('analysis_output', {})
                
                if chunk_id not in chunk_mapping:
                    continue
                    
                chunk_data = chunk_mapping[chunk_id]
                macro_chunk_id = chunk_data.get('Macro_Chunk_ID')
                speaker = chunk_data.get('Speaker')
                
                # 处理分析结果
                analyzed_Units = analysis_output.get('analyzed_Units', [])
                if isinstance(analyzed_Units, list):
                    unit_counter_in_chunk = 1
                    for u in analyzed_Units:
                        Unit_Text = safe_str_convert(u.get("Unit_Text", ""))
                        if not Unit_Text.strip():
                            continue
                            
                        unit_id = f"{macro_chunk_id}-{unit_counter_in_chunk}"
                        norm_text = normalize_text(Unit_Text)
                        Unit_hash = hashlib.sha256(norm_text.encode('utf-8')).hexdigest()
                        
                        final_data.append({
                            "Unit_ID": unit_id,
                            "Source": source,
                            "speaker": speaker, 
                            "Unit_Text": Unit_Text, 
                            "seed_sentence": u.get("seed_sentence", ""),
                            "expansion_logic": u.get("expansion_logic", ""), 
                            "Macro_Chunk_ID": macro_chunk_id,
                            "Unit_Hash": Unit_hash, 
                            "processing_status": ProcessingStatus.SUCCESS, 
                            "Incident": u.get("Incident", ""),
                            "Frame_ProblemDefinition": u.get("Frame_ProblemDefinition", []),
                            "Frame_ResponsibilityAttribution": u.get("Frame_ResponsibilityAttribution", []),
                            "Frame_MoralEvaluation": u.get("Frame_MoralEvaluation", []),
                            "Frame_SolutionRecommendation": u.get("Frame_SolutionRecommendation", []),
                            "Frame_ActionStatement": u.get("Frame_ActionStatement", []),
                            "Frame_CausalExplanation": u.get("Frame_CausalExplanation", []),
                            "Valence": u.get("Valence", ""), 
                            "Evidence_Type": u.get("Evidence_Type", ""),
                            "Attribution_Level": u.get("Attribution_Level", ""), 
                            "Temporal_Focus": u.get("Temporal_Focus", ""),
                            "Primary_Actor_Type": u.get("Primary_Actor_Type", ""), 
                            "Geographic_Scope": u.get("Geographic_Scope", ""),
                            "Relationship_Model_Definition": u.get("Relationship_Model_Definition", ""),
                            "Discourse_Type": u.get("Discourse_Type", "")
                        })
                        
                        unit_counter_in_chunk += 1
        else:
            # 批处理失败，记录失败信息
            for chunk_data in batch:
                macro_chunk_id = chunk_data.get('Macro_Chunk_ID')
                failed_unit = create_unified_record(
                    ProcessingStatus.API_FAILED, original_id, source, 
                    chunk_data.get('Macro_Chunk_Text', '')[:200], 
                    f"宏观块 {macro_chunk_id} 批处理分析失败"
                )
                failed_unit["Macro_Chunk_ID"] = macro_chunk_id
                final_data.append(failed_unit)
    
    return final_data

async def get_macro_chunks_media(row, api_service):
   full_text = safe_str_convert(row.get(COLUMN_MAPPING["MEDIA_TEXT"], ''))
   if not full_text.strip():
       return []
   language, config = detect_language_and_get_config(full_text)
   text_tokens = count_tokens(full_text)

   if text_tokens > config['MAX_SINGLE_TEXT']:
       UX.warn(f"文本过长({text_tokens} tokens, {len(full_text)}字符)，跳过")
       return None
   # 选择语言对应的模板
   template_map = {
       'ru': prompts.MACRO_CHUNKING_RU,
       'zh': prompts.MACRO_CHUNKING_ZH,
       'en': prompts.MACRO_CHUNKING_EN
   }
   macro_tpl = template_map.get(language, prompts.MACRO_CHUNKING_EN)
   try:
       prompt = macro_tpl.replace("{full_text}", full_text)
       macro_chunks = await api_service.get_analysis(
           prompt, 'macro_chunks', language,
           model_name=get_stage_model('MACRO_CHUNKING'),
           stage_key='MACRO_CHUNKING',
           context_label=f"ID={safe_str_convert(row.get(COLUMN_MAPPING['ID'], 'unknown'))}:MACRO"
       )
       if macro_chunks is None:
           return None

       if not isinstance(macro_chunks, list):
           # API返回格式异常，视为失败
           UX.warn(f"宏观块切分API返回格式异常，视为失败")
           return None
       return [chunk for chunk in macro_chunks if isinstance(chunk, dict)]

   except Exception as e:
       UX.warn(f"宏观分块失败: {e}")
       # 所有异常都视为失败，确保"全有或全无"
       return None

async def process_row(row_data, api_service, macro_db_path=None, output_file_path=None, macro_chunks_rerun=None, force_rechunk=False):
    """处理单行数据（最终修复版，使用永久ID）"""
    row = row_data[1]
    original_id = safe_str_convert(row.get(COLUMN_MAPPING["ID"]))
    source = row_data[2]
    
    failed_macro_chunk_ids = set()
    if macro_chunks_rerun and isinstance(macro_chunks_rerun, dict):
        failed_macro_chunk_ids = set(macro_chunks_rerun.get(original_id, set()))

    try:
        media_text = safe_str_convert(row.get(COLUMN_MAPPING["MEDIA_TEXT"], ''))
        article_title = safe_str_convert(row.get(COLUMN_MAPPING['MEDIA_TITLE'], '无标题'))
        macro_chunking_failed = bool(force_rechunk)
        
        macro_chunks = None
        is_reprocessing = False

        if macro_db_path and not macro_chunking_failed:
            macro_chunks = get_existing_macro_chunks(original_id, macro_db_path)
            if macro_chunks is not None:
                is_reprocessing = True
        
        if macro_chunks is None:
            if macro_chunking_failed: UX.info(f"ID {original_id}: 重新切分宏观块")
            macro_chunks = await get_macro_chunks_media(row, api_service)
            if macro_chunks: # 首次切分后，为新块赋予永久ID
                 for i, chunk in enumerate(macro_chunks):
                     chunk['Macro_Chunk_ID'] = f"{original_id}-M{i+1}"
                     chunk['Speaker'] = safe_get_speaker(chunk, '未知信源')
                     chunk['Macro_Chunk_Text'] = chunk.pop('macro_chunk_text', '')

        if macro_chunks is None:
            failed_record = create_unified_record(ProcessingStatus.API_FAILED, original_id, source, media_text, "宏观块切分失败")
            failed_record["Macro_Chunk_ID"] = f"{original_id}-MACRO_CHUNKING_FAILED"
            return original_id, [failed_record], None

        if not macro_chunks:
            no_relevant_record = create_unified_record(ProcessingStatus.NO_RELEVANT, original_id, source)
            no_relevant_record["Macro_Chunk_ID"] = f"{original_id}-NO_CONTENT"
            return original_id, [no_relevant_record], None

        macro_chunks_info = []
        if not is_reprocessing:
            for chunk in macro_chunks:
                macro_chunks_info.append({
                    'Macro_Chunk_ID': chunk['Macro_Chunk_ID'], 'Original_ID': original_id,
                    'Source': source, 'Article_Title': article_title,
                    'Speaker': chunk['Speaker'], 'Macro_Chunk_Text': chunk['Macro_Chunk_Text']
                })
        
        chunks_to_process = []
        if not failed_macro_chunk_ids: 
            chunks_to_process = macro_chunks
        else:
            for chunk in macro_chunks:
                if chunk.get('Macro_Chunk_ID') in failed_macro_chunk_ids:
                    chunks_to_process.append(chunk)
        
        if not chunks_to_process:
             UX.ok(f"✅ ID {original_id}: 本次运行无需处理新的宏观块。")
             return original_id, [], None
        
        UX.info(f"ID {original_id}: 计划处理 {len(chunks_to_process)} 个宏观块。")
        final_data = []  # 直接构建最终数据列表

        # 根据配置和宏观块长度选择处理方式
        if ENABLE_BATCH_PROCESSING and len(chunks_to_process) > 1:
            # 将宏观块按长度分组：超短块批处理，其他单独处理
            short_chunks = []
            normal_chunks = []
            
            for chunk_data in chunks_to_process:
                chunk_text = chunk_data.get('Macro_Chunk_Text', '').strip()
                if not chunk_text:
                    continue
                    
                token_count = count_tokens(chunk_text)
                if token_count < SHORT_TEXT_THRESHOLD:
                    short_chunks.append(chunk_data)
                else:
                    normal_chunks.append(chunk_data)
            
            UX.info(f"ID {original_id}: 识别到 {len(short_chunks)} 个超短块（批处理），{len(normal_chunks)} 个正常块（单独处理）")
            
            # 处理超短块（批处理）
            if short_chunks:
                UX.info(f"ID {original_id}: 批处理 {len(short_chunks)} 个超短块，批次大小: {BATCH_SIZE}")
                batch_data = await process_chunks_batch(short_chunks, original_id, source, api_service)
                final_data.extend(batch_data)
            
            # 处理正常/长块（单独处理）
            if normal_chunks:
                UX.info(f"ID {original_id}: 单独处理 {len(normal_chunks)} 个正常/长块")
                for chunk_data in normal_chunks:
                    chunk_text = chunk_data.get('Macro_Chunk_Text', '').strip()
                    if not chunk_text: continue

                    macro_chunk_id = chunk_data.get('Macro_Chunk_ID')
                    speaker = chunk_data.get('Speaker')
                    token_count = count_tokens(chunk_text)
                    language, _ = detect_language_and_get_config(chunk_text)

                    analysis_model_key = 'INTEGRATED_ANALYSIS'
                    if token_count < SHORT_TEXT_THRESHOLD: analysis_model_key = 'INTEGRATED_ANALYSIS_SHORT'
                    elif token_count > LONG_TEXT_THRESHOLD: analysis_model_key = 'INTEGRATED_ANALYSIS_LONG'
                    
                    integrated_prompt = prompts.INTEGRATED_ANALYSIS_V2.replace("{speaker}", speaker).replace("{macro_chunk_text}", chunk_text)
                    
                    analyzed_Units = await api_service.get_analysis(
                        integrated_prompt, 'analyzed_Units', language,
                        model_name=get_stage_model(analysis_model_key), stage_key=analysis_model_key,
                        context_label=f"{macro_chunk_id}:{analysis_model_key}"
                    )
                    
                    if analyzed_Units is None:
                        failed_unit = create_unified_record(ProcessingStatus.API_FAILED, original_id, source, chunk_text[:200], f"宏观块 {macro_chunk_id} 分析失败")
                        failed_unit["Macro_Chunk_ID"] = macro_chunk_id
                        # 失败记录直接添加，它的Unit_ID由create_unified_record内部定义
                        final_data.append(failed_unit)
                        continue
                    
                    if isinstance(analyzed_Units, list):
                        # 如果API返回空列表，则不执行任何操作，自然地实现了"良性过滤"
                        
                        # [核心修改] 在宏观块内部初始化序号计数器
                        unit_counter_in_chunk = 1
                        for u in analyzed_Units:
                            Unit_Text = safe_str_convert(u.get("Unit_Text", ""))
                            if not Unit_Text.strip():
                                # 您处理空内容的逻辑，也直接在这里添加
                                failed_unit = create_unified_record(
                                    ProcessingStatus.API_FAILED, 
                                    original_id, 
                                    source, 
                                    chunk_text[:200], 
                                    f"宏观块 {macro_chunk_id} 返回空内容"
                                )
                                failed_unit["Macro_Chunk_ID"] = macro_chunk_id
                                final_data.append(failed_unit)
                                continue
                            
                            # 构建稳定、分层的Unit_ID
                            unit_id = f"{macro_chunk_id}-{unit_counter_in_chunk}"
                            
                            # 准备并直接添加完整的单元数据到 final_data
                            norm_text = normalize_text(Unit_Text)
                            Unit_hash = hashlib.sha256(norm_text.encode('utf-8')).hexdigest()
                            
                            final_data.append({
                                "Unit_ID": unit_id,
                                "Source": source,
                                "speaker": speaker, 
                                "Unit_Text": Unit_Text, 
                                "seed_sentence": u.get("seed_sentence", ""),
                                "expansion_logic": u.get("expansion_logic", ""), 
                                "Macro_Chunk_ID": macro_chunk_id,
                                "Unit_Hash": Unit_hash, 
                                "processing_status": ProcessingStatus.SUCCESS, 
                                "Incident": u.get("Incident", ""),
                                "Frame_ProblemDefinition": u.get("Frame_ProblemDefinition", []),
                                "Frame_ResponsibilityAttribution": u.get("Frame_ResponsibilityAttribution", []),
                                "Frame_MoralEvaluation": u.get("Frame_MoralEvaluation", []),
                                "Frame_SolutionRecommendation": u.get("Frame_SolutionRecommendation", []),
                                "Frame_ActionStatement": u.get("Frame_ActionStatement", []),
                                "Frame_CausalExplanation": u.get("Frame_CausalExplanation", []),
                                "Valence": u.get("Valence", ""), 
                                "Evidence_Type": u.get("Evidence_Type", ""),
                                "Attribution_Level": u.get("Attribution_Level", ""), 
                                "Temporal_Focus": u.get("Temporal_Focus", ""),
                                "Primary_Actor_Type": u.get("Primary_Actor_Type", ""), 
                                "Geographic_Scope": u.get("Geographic_Scope", ""),
                                "Relationship_Model_Definition": u.get("Relationship_Model_Definition", ""),
                                "Discourse_Type": u.get("Discourse_Type", "")
                            })
                            
                            # 序号递增
                            unit_counter_in_chunk += 1
        else:
            # 完全禁用批处理，所有宏观块单独处理
            UX.info(f"ID {original_id}: 批处理已禁用或只有单个宏观块，使用单个处理模式")
            for chunk_data in chunks_to_process:
                chunk_text = chunk_data.get('Macro_Chunk_Text', '').strip()
                if not chunk_text: continue

                macro_chunk_id = chunk_data.get('Macro_Chunk_ID')
                speaker = chunk_data.get('Speaker')
                token_count = count_tokens(chunk_text)
                language, _ = detect_language_and_get_config(chunk_text)

                analysis_model_key = 'INTEGRATED_ANALYSIS'
                if token_count < SHORT_TEXT_THRESHOLD: analysis_model_key = 'INTEGRATED_ANALYSIS_SHORT'
                elif token_count > LONG_TEXT_THRESHOLD: analysis_model_key = 'INTEGRATED_ANALYSIS_LONG'
                
                integrated_prompt = prompts.INTEGRATED_ANALYSIS_V2.replace("{speaker}", speaker).replace("{macro_chunk_text}", chunk_text)
                
                analyzed_Units = await api_service.get_analysis(
                    integrated_prompt, 'analyzed_Units', language,
                    model_name=get_stage_model(analysis_model_key), stage_key=analysis_model_key,
                    context_label=f"{macro_chunk_id}:{analysis_model_key}"
                )
                
                if analyzed_Units is None:
                    failed_unit = create_unified_record(ProcessingStatus.API_FAILED, original_id, source, chunk_text[:200], f"宏观块 {macro_chunk_id} 分析失败")
                    failed_unit["Macro_Chunk_ID"] = macro_chunk_id
                    final_data.append(failed_unit)
                    continue
                
                if isinstance(analyzed_Units, list):
                    unit_counter_in_chunk = 1
                    for u in analyzed_Units:
                        Unit_Text = safe_str_convert(u.get("Unit_Text", ""))
                        if not Unit_Text.strip():
                            failed_unit = create_unified_record(
                                ProcessingStatus.API_FAILED, 
                                original_id, 
                                source, 
                                chunk_text[:200], 
                                f"宏观块 {macro_chunk_id} 返回空内容"
                            )
                            failed_unit["Macro_Chunk_ID"] = macro_chunk_id
                            final_data.append(failed_unit)
                            continue
                        
                        unit_id = f"{macro_chunk_id}-{unit_counter_in_chunk}"
                        norm_text = normalize_text(Unit_Text)
                        Unit_hash = hashlib.sha256(norm_text.encode('utf-8')).hexdigest()
                        
                        final_data.append({
                            "Unit_ID": unit_id,
                            "Source": source,
                            "speaker": speaker, 
                            "Unit_Text": Unit_Text, 
                            "seed_sentence": u.get("seed_sentence", ""),
                            "expansion_logic": u.get("expansion_logic", ""), 
                            "Macro_Chunk_ID": macro_chunk_id,
                            "Unit_Hash": Unit_hash, 
                            "processing_status": ProcessingStatus.SUCCESS, 
                            "Incident": u.get("Incident", ""),
                            "Frame_ProblemDefinition": u.get("Frame_ProblemDefinition", []),
                            "Frame_ResponsibilityAttribution": u.get("Frame_ResponsibilityAttribution", []),
                            "Frame_MoralEvaluation": u.get("Frame_MoralEvaluation", []),
                            "Frame_SolutionRecommendation": u.get("Frame_SolutionRecommendation", []),
                            "Frame_ActionStatement": u.get("Frame_ActionStatement", []),
                            "Frame_CausalExplanation": u.get("Frame_CausalExplanation", []),
                            "Valence": u.get("Valence", ""), 
                            "Evidence_Type": u.get("Evidence_Type", ""),
                            "Attribution_Level": u.get("Attribution_Level", ""), 
                            "Temporal_Focus": u.get("Temporal_Focus", ""),
                            "Primary_Actor_Type": u.get("Primary_Actor_Type", ""), 
                            "Geographic_Scope": u.get("Geographic_Scope", ""),
                            "Relationship_Model_Definition": u.get("Relationship_Model_Definition", ""),
                            "Discourse_Type": u.get("Discourse_Type", "")
                        })
                        
                        unit_counter_in_chunk += 1

        if not final_data:
            no_relevant_record = create_unified_record(ProcessingStatus.NO_RELEVANT, original_id, source)
            if failed_macro_chunk_ids:
                 no_relevant_record['reprocessed_chunks'] = list(failed_macro_chunk_ids)
            return original_id, [no_relevant_record], macro_chunks_info

        return original_id, final_data, macro_chunks_info

    except Exception as e:
        UX.err(f"处理ID {original_id} 时发生错误: {e}")
        return original_id, [create_unified_record(ProcessingStatus.API_FAILED, original_id, source, "", f"错误: {str(e)[:100]}")], None

async def api_worker(name, task_queue, results_queue, api_service, source, pbar, macro_db_path=None, output_file_path=None, macro_chunks_rerun=None, rechunk_article_ids=None):
   while True:
       try:
           item = await task_queue.get()
           if item is None:
               break
           
           try:
               item_with_source = (item[0], item[1], source)
               current_id = safe_str_convert(item[1].get(COLUMN_MAPPING["ID"], "unknown")) if len(item) > 1 else "unknown"
               force_rechunk = (rechunk_article_ids is not None and current_id in rechunk_article_ids)
               original_id, result_Units, macro_chunks_info = await process_row(
                   item_with_source,
                   api_service,
                   macro_db_path,
                   output_file_path,
                   macro_chunks_rerun=macro_chunks_rerun,
                   force_rechunk=force_rechunk
               )
               await results_queue.put((original_id, result_Units, macro_chunks_info))
           except Exception as e:
               UX.warn(f"工作者 {name} 处理失败: {e}")
               original_id = safe_str_convert(item[1].get(COLUMN_MAPPING["ID"], "unknown")) if len(item) > 1 else "unknown"
               await results_queue.put((original_id, [create_unified_record(ProcessingStatus.API_FAILED, original_id, source, "", f"Worker错误: {str(e)[:100]}")], None))
           finally:
               # 保证无论成功失败，任务完成后都更新进度条
               pbar.update(1)
               task_queue.task_done()
       
       except asyncio.CancelledError:
           break
       except Exception as e:
           UX.err(f"工作者 {name} 发生严重错误: {e}")

async def saver_worker(results_queue, df_input, output_file_path, macro_db_path, total_tasks=None):
    """
    [优化后] 批量保存worker：
    - 缓冲并分批保存"分析结果"
    - 缓冲并分批保存"宏观块"到主数据库
    """
    analysis_buffer = []
    macro_chunk_buffer = []
    received_count = 0
    
    # 从配置读取缓冲区大小阈值
    ANALYSIS_BUFFER_LIMIT = BUFFER_CONFIG.get('analysis_buffer_limit', 30)    # 分析结果缓冲区
    MACRO_CHUNK_BUFFER_LIMIT = BUFFER_CONFIG.get('macro_chunk_buffer_limit', 80)  # 宏观块缓冲区

    async def save_analysis_batch(data):
        if not data:
            return
        async with file_write_lock:
            # 调用现有的分析结果保存函数
            save_data_to_excel(data, df_input, output_file_path)

    async def save_macro_batch(data):
        if not data:
            return
        async with file_write_lock:
            # 调用新增的宏观块数据库保存函数
            save_macro_chunks_database(data, macro_db_path)

    while True:
        try:
            # 等待结果，从配置读取超时时间
            item = await asyncio.wait_for(results_queue.get(), timeout=QUEUE_TIMEOUT)
            if item is None: # 收到结束信号
                await save_analysis_batch(analysis_buffer)
                await save_macro_batch(macro_chunk_buffer)
                break

            original_id, result_Units, macro_chunks_info = item
            received_count += 1

            # 收集宏观块到缓冲区
            if macro_chunks_info:
                macro_chunk_buffer.extend(macro_chunks_info)

            # 收集分析结果到缓冲区
            if result_Units:
                for unit in result_Units:
                    unit[COLUMN_MAPPING["ID"]] = original_id
                analysis_buffer.extend(result_Units)

            # 检查是否需要清空并保存缓冲区
            if len(analysis_buffer) >= ANALYSIS_BUFFER_LIMIT:
                await save_analysis_batch(analysis_buffer)
                UX.info(f"批量保存分析结果: {len(analysis_buffer)} 条")
                analysis_buffer = []

            if len(macro_chunk_buffer) >= MACRO_CHUNK_BUFFER_LIMIT:
                await save_macro_batch(macro_chunk_buffer)
                UX.info(f"批量保存宏观块: {len(macro_chunk_buffer)} 条")
                macro_chunk_buffer = []

            results_queue.task_done()
            if total_tasks and received_count >= total_tasks:
                await save_analysis_batch(analysis_buffer)
                await save_macro_batch(macro_chunk_buffer)
                break

        except asyncio.TimeoutError:
            # 超时后保存所有缓冲区内容，防止在任务间隙丢失数据
            await save_analysis_batch(analysis_buffer)
            analysis_buffer = []
            await save_macro_batch(macro_chunk_buffer)
            macro_chunk_buffer = []
            if total_tasks and received_count >= total_tasks:
                break
            continue

        except asyncio.CancelledError:
            await save_analysis_batch(analysis_buffer)
            await save_macro_batch(macro_chunk_buffer)
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

           input_base_cols = [col for col in df_input.columns if col in df_new_Units.columns and col != COLUMN_MAPPING["ID"]]
           if input_base_cols:
               df_new_Units = df_new_Units.drop(columns=input_base_cols)
           
           # [最终修复] 全新的、更健壮的清理逻辑
           if not df_existing.empty and 'Macro_Chunk_ID' in df_existing.columns:
               chunks_to_clean_up = set()
               
               # 1. 从成功和失败的新记录中，直接获取宏观块ID
               if 'Macro_Chunk_ID' in df_new_Units.columns:
                   chunks_to_clean_up.update(df_new_Units['Macro_Chunk_ID'].dropna().astype(str))

               # 2. 从NO_RELEVANT记录的"便签"中获取需要清理的旧宏观块ID
               if 'reprocessed_chunks' in df_new_Units.columns:
                   no_relevant_rows = df_new_Units[df_new_Units['processing_status'] == ProcessingStatus.NO_RELEVANT]
                   if not no_relevant_rows.empty:
                       for chunk_list in no_relevant_rows['reprocessed_chunks'].dropna():
                           if isinstance(chunk_list, list):
                               chunks_to_clean_up.update(chunk_list)
               
               if chunks_to_clean_up:
                   # 执行清理
                   initial_rows = len(df_existing)
                   df_existing = df_existing[~df_existing['Macro_Chunk_ID'].astype(str).isin(chunks_to_clean_up)]
                   rows_removed = initial_rows - len(df_existing)
                   if rows_removed > 0:
                       UX.info(f"数据清理：移除了 {rows_removed} 条与本次重处理相关的旧记录。")
           
           df_input_subset = df_input[df_input[COLUMN_MAPPING["ID"]].isin(new_original_ids)].copy()
           df_newly_merged = pd.merge(df_input_subset, df_new_Units, on=COLUMN_MAPPING["ID"], how='left')
       else:
           df_newly_merged = df_new_Units.copy()

       df_final_to_save = pd.concat([df_existing, df_newly_merged], ignore_index=True)

       if 'Unit_ID' in df_final_to_save.columns:
           df_final_to_save = df_final_to_save.drop_duplicates(subset=['Unit_ID'], keep='last')

       df_final_to_save = reorder_columns(df_final_to_save, list(df_input.columns))

       if 'reprocessed_chunks' in df_final_to_save.columns:
           df_final_to_save = df_final_to_save.drop(columns=['reprocessed_chunks'])

       df_final_to_save.to_excel(output_file_path, index=False)
       UX.ok(f"已保存: {os.path.basename(output_file_path)} (累计 {len(df_final_to_save)} 条)")

   except Exception as e:
       UX.err(f"Excel保存失败: {e}")

async def main_async():
   UX.start_run()
   UX.phase("长文本分析器启动")
   UX.info(f"信度检验模式: {'开启' if RELIABILITY_TEST_MODE else '关闭'}")

   # 验证模型配置
   required_keys = ["MACRO_CHUNKING", "INTEGRATED_ANALYSIS", "INTEGRATED_ANALYSIS_SHORT"]
   # 检查新的model_pools配置
   primary_models = MODEL_POOLS.get('primary_models', {})
   missing = [k for k in required_keys if k not in primary_models]
   if missing:
       raise ValueError(f"缺少模型池配置: {missing}")

   # 检查API密钥
   if not API_CONFIG["API_KEYS"] or not API_CONFIG["API_KEYS"][0]:
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
       api_service = APIService(session)

       for file_path in files_to_process:
           file_basename = os.path.basename(file_path)
           UX.phase(f"处理文件: {file_basename}")

           # 识别信源
           source = identify_source(file_basename)
           UX.info(f"识别信源: {source}")

           output_file_path = os.path.join(OUTPUT_PATH, f"(不能删)analyzed_{file_basename}") if is_folder_mode else OUTPUT_PATH

           # 读取输入文件
           try:
               df_input = pd.read_excel(file_path)
               if COLUMN_MAPPING["MEDIA_TEXT"] not in df_input.columns:
                   UX.err("文件缺少必要的文本列")
                   continue
           except Exception as e:
               UX.err(f"读取文件失败: {e}")
               continue

           # 定义宏观块数据库路径
           macro_db_path = os.path.join(OUTPUT_PATH, '(不能删)媒体_宏观块主数据库.xlsx')

           # 🔍 构建断点续传计划
           UX.info("🔍 构建断点续传计划...")
           total_input_articles = len(set(df_input[COLUMN_MAPPING["ID"]].astype(str)))
           never_processed_ids, rechunk_article_ids, macro_chunks_to_rerun = build_resume_plan(
               output_file_path, df_input, COLUMN_MAPPING["ID"]
           )

           ids_to_process = set(never_processed_ids) | set(rechunk_article_ids) | set(macro_chunks_to_rerun.keys())

           UX.info(f"📊 计划：总文章数 {total_input_articles}")
           UX.info(f"   从未处理: {len(never_processed_ids)} 篇")
           UX.info(f"   需重切分: {len(rechunk_article_ids)} 篇")
           rerun_chunks_count = sum(len(v) for v in macro_chunks_to_rerun.values())
           UX.info(f"   宏观块重分析: {rerun_chunks_count} 个，涉及 {len(macro_chunks_to_rerun)} 篇")
           UX.info(f"   本次处理文章: {len(ids_to_process)} 篇，跳过 {total_input_articles - len(ids_to_process)} 篇")

           df_to_process = df_input[df_input[COLUMN_MAPPING["ID"]].astype(str).isin(ids_to_process)].copy()

           if len(df_to_process) > 0:
               UX.info(f"🚀 KISS断点续传: 处理{len(df_to_process)}篇文章 (process_row将自动跳过已成功的宏观块)")

               # 创建任务队列
               task_queue = asyncio.Queue()
               results_queue = asyncio.Queue()
               total_tasks = len(df_to_process)

               # 添加任务到队列
               for item in df_to_process.iterrows():
                   await task_queue.put(item)

               # 创建进度条
               pbar = aio_tqdm(total=total_tasks, desc=f"处理中 ({file_basename})")

               # [修改] 创建保存任务 - 传入宏观块数据库路径
               saver_task = asyncio.create_task(
                   saver_worker(results_queue, df_input, output_file_path, macro_db_path, total_tasks)
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
                           pbar, # <-- 将进度条对象传递给worker
                           macro_db_path,
                           output_file_path,
                           macro_chunks_rerun=macro_chunks_to_rerun,
                           rechunk_article_ids=rechunk_article_ids
                       )
                   )
                   for i in range(MAX_CONCURRENT_REQUESTS)
               ]

               # 等待所有任务都被worker处理完毕
               await task_queue.join()
               
               pbar.close()
           
           elif len(ids_to_process) == 0:
               UX.ok("🎉 该文件所有条目已完美处理完毕！")
               continue
           
           # 🎉 处理完成总结
           try:
               df_final_check = pd.read_excel(output_file_path)
               if not df_final_check.empty and 'processing_status' in df_final_check.columns:
                   final_success = (df_final_check['processing_status'] == ProcessingStatus.SUCCESS).sum()
                   final_no_relevant = (df_final_check['processing_status'] == ProcessingStatus.NO_RELEVANT).sum()
                   final_failed = (df_final_check['processing_status'] == ProcessingStatus.API_FAILED).sum()
                   final_total_units = len(df_final_check)
                   
                   # 计算文章级别完成度
                   final_processed_ids, final_failed_ids = get_processing_state(df_final_check, COLUMN_MAPPING["ID"])
                   final_completed_articles = len(final_processed_ids - final_failed_ids)
                   final_completion_rate = (final_completed_articles / max(1, total_input_articles)) * 100
                   
                   UX.ok(f"📋 文件 {file_basename} 处理完成总结:")
                   UX.ok(f"   📊 文章完成度: {final_completed_articles}/{total_input_articles} ({final_completion_rate:.1f}%)")
                   UX.ok(f"   🎯 议题单元: 成功{final_success}条, 无相关{final_no_relevant}条, 失败{final_failed}条")
                   
                   if final_failed_ids:
                       UX.warn(f"   ⚠️  仍有{len(final_failed_ids)}篇文章处理失败，可再次运行进行智能重试")
                   else:
                       UX.ok(f"   ✨ 完美！所有文章均已成功处理")
               else:
                   UX.ok(f"文件 {file_basename} 处理完成")
           except Exception:
               UX.ok(f"文件 {file_basename} 处理完成")

   # 生成信度检验文件
   if RELIABILITY_TEST_MODE:
       UX.phase("生成信度检验文件")

       # 宏观块数据库路径（现在由实时保存生成）
       macro_db_path = os.path.join(OUTPUT_PATH, '(不能删)媒体_宏观块主数据库.xlsx')

       # 合并所有结果文件
       final_results_files = glob.glob(os.path.join(OUTPUT_PATH, "*analyzed_*.xlsx"))
       if final_results_files:
           all_results = []
           for file in final_results_files:
               df = pd.read_excel(file)
               df = ensure_required_columns(df)
               source = identify_source(os.path.basename(file))
               df['Source'] = source
               all_results.append(df)

           if all_results:
               df_all_results = pd.concat(all_results, ignore_index=True)
               df_all_results = ensure_required_columns(df_all_results)
               df_all_results = reorder_columns(df_all_results, [])

               combined_results_path = os.path.join(OUTPUT_PATH, '媒体_最终分析数据库.xlsx')
               df_all_results.to_excel(combined_results_path, index=False)

               if 'processing_status' in df_all_results.columns:
                   success_count = (df_all_results['processing_status'] == ProcessingStatus.SUCCESS).sum()
                   no_relevant_count = (df_all_results['processing_status'] == ProcessingStatus.NO_RELEVANT).sum()
                   failed_count = (df_all_results['processing_status'] == ProcessingStatus.API_FAILED).sum()
                   UX.ok(f"最终数据库已保存: {combined_results_path}")
                   UX.info(f"总记录: {len(df_all_results)}, 成功: {success_count}, 无相关: {no_relevant_count}, 失败: {failed_count}")

               # 生成信度检验文件 - 使用正确的宏观块数据
               if macro_db_path and os.path.exists(macro_db_path):
                   generate_reliability_files_cn(macro_db_path, combined_results_path, OUTPUT_PATH)
               else:
                   UX.err("宏观块数据库不存在，无法生成信度检验文件")

           else:
               UX.warn("合并结果文件失败")
       else:
           UX.warn("未找到任何analyzed结果文件")

   UX.phase("所有任务完成")

# ============================================================================
# 信度检验核心修复代码
# ============================================================================

def _highlight_Unit_in_parent(parent_text: str, Unit_Text: str) -> str:
    if not parent_text or not Unit_Text:
        return parent_text

    parent_str = str(parent_text).strip()
    unit_str = str(Unit_Text).strip()

    # 规范化空格用于比较
    def normalize(s):
        return ' '.join(s.split())

    parent_norm = normalize(parent_str)
    unit_norm = normalize(unit_str)

    # 情况1：议题单元就是整个宏观块
    if parent_norm == unit_norm or len(parent_norm) - len(unit_norm) < 10:
        return f"【{parent_str}】"

    # 情况2：议题单元是宏观块的一部分
    # 由于议题单元是从宏观块提取的，理论上一定能找到
    # 尝试直接查找
    if unit_str in parent_str:
        return parent_str.replace(unit_str, f"【{unit_str}】", 1)

    # 处理可能的空格/换行差异
    if unit_norm in parent_norm:
        # 找到归一化后的位置
        idx = parent_norm.find(unit_norm)
        # 在原文中找到对应位置（考虑空格差异）
        words_before = len(parent_norm[:idx].split())
        # 在原文中定位
        words = parent_str.split()
        if words_before < len(words):
            # 重建高亮文本
            result_parts = []
            word_count = 0
            in_highlight = False
            unit_words = unit_norm.split()

            for word in parent_str.split():
                if word_count == words_before and not in_highlight:
                    result_parts.append("【")
                    in_highlight = True

                result_parts.append(word)

                if in_highlight:
                    if word_count >= words_before + len(unit_words) - 1:
                        result_parts.append("】")
                        in_highlight = False

                word_count += 1

            return ' '.join(result_parts)

    # 最后的备选：模糊匹配
    # 找到最相似的片段并高亮
    import difflib

    # 将文本分句
    sentences = [s.strip() for s in re.split(r'[。！？.!?\n]', parent_str) if s.strip()]
    unit_sentences = [s.strip() for s in re.split(r'[。！？.!?\n]', unit_str) if s.strip()]

    if unit_sentences:
        # 找到包含议题单元第一句的位置
        first_unit_sentence = unit_sentences[0]
        best_match_idx = -1
        best_ratio = 0

        for i, sent in enumerate(sentences):
            ratio = difflib.SequenceMatcher(None, sent, first_unit_sentence).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match_idx = i

        similarity_threshold = QUALITY_THRESHOLDS.get('text_similarity_threshold', 0.7)

        if best_match_idx >= 0 and best_ratio > similarity_threshold:
            # 从找到的位置开始高亮相应长度的文本
            highlighted_sentences = sentences.copy()
            num_sentences_to_highlight = len(unit_sentences)

            for j in range(best_match_idx, min(best_match_idx + num_sentences_to_highlight, len(sentences))):
                highlighted_sentences[j] = f"【{sentences[j]}】"

            # 重组文本
            result = ""
            for sent in highlighted_sentences:
                if sent:
                    result += sent + "。"

            return result.rstrip("。")

    # 如果实在找不到（不应该发生），返回带标记的原文
    UX.warn(f"警告：无法在宏观块中定位议题单元")
    return f"{parent_str}\n\n[议题单元：【{unit_str}】]"

# 本地化文本缓存
_locale_cache = {}

def _load_locale_mapping(lang: str) -> dict:
    """从JSON文件加载本地化映射"""
    if lang in _locale_cache:
        return _locale_cache[lang]

    locales_dir = os.path.join(os.path.dirname(__file__), 'locales')
    file_path = os.path.join(locales_dir, f'{lang}.json')

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
            _locale_cache[lang] = mapping
            return mapping
    except Exception as e:
        print(f"加载本地化文件 {lang}.json 失败: {e}")
        return {}

def _decorate_headers(df: pd.DataFrame, lang: str) -> pd.DataFrame:
    """装饰列名为语言标签格式"""
    mapping = _load_locale_mapping(lang)
    new_cols = []

    for c in list(df.columns):
        if c in mapping:
            new_cols.append(f"{mapping[c]}({c})")
        else:
            new_cols.append(c)

    df_out = df.copy()
    df_out.columns = new_cols
    return df_out

def _save_bilingual(df: pd.DataFrame, zh_path: str, ru_path: str):
    """保存双语版本文件"""
    try:
        zh_dir = os.path.dirname(zh_path)
        ru_dir = os.path.dirname(ru_path)

        if zh_dir:
            os.makedirs(zh_dir, exist_ok=True)

        if ru_dir and ru_dir != zh_dir:
            os.makedirs(ru_dir, exist_ok=True)
    except Exception as e:
        UX.warn(f"创建输出目录失败: {e}")

    try:
        df_zh = _decorate_headers(df, 'zh')
        df_zh.to_excel(zh_path, index=False)
    except Exception as e:
        UX.warn(f"中文版本导出失败: {e}")

    try:
        df_ru = _decorate_headers(df, 'ru')
        df_ru.to_excel(ru_path, index=False)
    except Exception as e:
        UX.warn(f"俄语版本导出失败: {e}")

def generate_reliability_files_cn(macro_db_path: str, final_results_path: str, output_path: str):
    """生成信度检验文件 - 不再尝试错误的重建"""
    # 检查输入文件
    if not os.path.exists(macro_db_path):
        UX.err(f"宏观块数据库文件不存在: {macro_db_path}")
        UX.err("请确保程序正确保存了宏观块数据")
        return

    if not os.path.exists(final_results_path):
        UX.err(f"最终结果文件不存在: {final_results_path}")
        return

    try:
        os.makedirs(output_path, exist_ok=True)
    except Exception as e:
        UX.warn(f"创建信度输出目录失败: {e}")
        return

    UX.info("开始生成信度检验文件（中文+俄语双语版）...")

    # 读取数据
    try:
        df_macro = pd.read_excel(macro_db_path)
        df_results = pd.read_excel(final_results_path)

        UX.info(f"加载宏观块数据: {len(df_macro)} 条")
        UX.info(f"加载分析结果: {len(df_results)} 条")

        # 验证数据完整性
        if 'Macro_Chunk_Text' not in df_macro.columns:
            UX.err("宏观块数据缺少Macro_Chunk_Text列 - 数据不完整")
            return

        # 检查宏观块文本长度（字符数和token数）
        macro_texts = df_macro['Macro_Chunk_Text'].astype(str)
        avg_char_len = macro_texts.str.len().mean()
        avg_token_len = macro_texts.apply(count_tokens).mean()

        UX.info(f"宏观块平均长度: {avg_char_len:.0f} 字符, {avg_token_len:.0f} tokens")

        min_chars_threshold = QUALITY_THRESHOLDS.get('min_macro_chunk_chars', 50)
        if avg_char_len < min_chars_threshold:
            UX.warn(f"宏观块文本异常短（平均{avg_char_len:.0f}字符 < {min_chars_threshold}字符），可能存在数据问题")

    except Exception as e:
        UX.err(f"读取文件失败: {e}")
        return

    # =========== 1. 读取和准备数据 ===========
    import pickle

    sampled_cache = os.path.join(output_path, '.sampled_ids.pkl')
    sampled_records = {'recall': set(), 'precision': set()}

    if os.path.exists(sampled_cache):
        try:
            with open(sampled_cache, 'rb') as f:
                loaded = pickle.load(f)
                if isinstance(loaded, dict):
                    sampled_records.update(loaded)
                    UX.info(f"读取已抽样记录: 召回{len(sampled_records.get('recall', set()))}条, "
                           f"精确{len(sampled_records.get('precision', set()))}条")
        except Exception as e:
            UX.warn(f"读取抽样记录失败: {e}")

    # 检查必要列是否存在
    if 'Macro_Chunk_ID' not in df_macro.columns:
        UX.err("宏观块数据缺少Macro_Chunk_ID列")
        return

    if 'processing_status' not in df_results.columns:
        UX.warn("结果数据缺少processing_status列，尝试兼容旧版本")
        # 尝试通过speaker字段判断状态
        if 'speaker' in df_results.columns:
            df_results['processing_status'] = df_results.apply(
                lambda x: ProcessingStatus.API_FAILED if 'API_CALL_FAILED' in str(x.get('speaker', ''))
                else ProcessingStatus.SUCCESS, axis=1
            )
        else:
            UX.err("无法确定处理状态")
            return

    # 标记使用情况
    used_macro_ids = set()
    excluded_original_ids = set()  # 被判定为NO_RELEVANT的原始ID

    if 'processing_status' in df_results.columns:
        # 找出成功处理的宏观块
        success_mask = df_results['processing_status'] == ProcessingStatus.SUCCESS
        if 'Macro_Chunk_ID' in df_results.columns:
            used_macro_ids = set(df_results[success_mask]['Macro_Chunk_ID'].dropna().astype(str).unique())

        # 找出被判定为NO_RELEVANT的原始文章ID
        no_relevant_mask = df_results['processing_status'] == ProcessingStatus.NO_RELEVANT
        if no_relevant_mask.any() and COLUMN_MAPPING["ID"] in df_results.columns:
            excluded_original_ids = set(df_results[no_relevant_mask][COLUMN_MAPPING["ID"]].dropna().astype(str).unique())

    # 在宏观块数据中标记
    df_macro['Was_Used'] = df_macro['Macro_Chunk_ID'].astype(str).isin(used_macro_ids)
    df_macro['Article_Was_Excluded'] = df_macro['Original_ID'].astype(str).isin(excluded_original_ids)

    # =========== 2. 反向检验（召回率）===========
    UX.info("生成反向检验样本...")

    negative_samples = []
    new_recall_ids = set()

    for source, cfg in RELIABILITY_SAMPLING_CONFIG.items():
        # 选择未使用或被排除的宏观块
        mask_source = df_macro['Source'] == source
        mask_not_used = (~df_macro['Was_Used']) | df_macro['Article_Was_Excluded']
        mask_not_sampled = ~df_macro['Macro_Chunk_ID'].astype(str).isin(sampled_records['recall'])

        candidates = df_macro[mask_source & mask_not_used & mask_not_sampled]
        n_sample = min(int(cfg.get('recall', 0)), len(candidates))

        if n_sample > 0:
            random_seed = RANDOMIZATION_CONFIG.get('random_seed', 2025)
            sample_df = candidates.sample(n=n_sample, replace=False, random_state=random_seed)
            negative_samples.append(sample_df)
            new_recall_ids.update(sample_df['Macro_Chunk_ID'].astype(str).tolist())

    if negative_samples:
        df_neg = pd.concat(negative_samples, ignore_index=True)
        df_neg = df_neg.drop_duplicates(subset=['Macro_Chunk_ID'], keep='first')

        # 构建输出数据
        df_neg_out = pd.DataFrame()
        df_neg_out['Macro_Chunk_ID'] = df_neg['Macro_Chunk_ID']
        df_neg_out['Source'] = df_neg['Source']
        df_neg_out['Original_ID'] = df_neg['Original_ID']
        df_neg_out['Article_Title'] = df_neg.get('Article_Title', '')

        # 添加AI决策标记
        df_neg_out['AI_Decision'] = df_neg.apply(
            lambda x: 'AI判定整篇文章无关' if x['Article_Was_Excluded'] else '未被使用的宏观块',
            axis=1
        )

        df_neg_out['Speaker'] = df_neg['Speaker']
        df_neg_out['Macro_Chunk_Text'] = df_neg['Macro_Chunk_Text']

        # 检验员字段
        df_neg_out['Inspector_Is_CN_RU_Related'] = ''  # 是/否
        df_neg_out['Inspector_Should_Include'] = ''    # 应该包含/不应该包含
        df_neg_out['Inspector_Comments'] = ''           # 备注

        # 保存文件
        zh_neg = os.path.join(output_path, '反向检验_召回率样本.xlsx')
        ru_neg = os.path.join(output_path, 'Проверка_отрицательная_выборка(Recall).xlsx')
        _save_bilingual(df_neg_out, zh_neg, ru_neg)

        UX.ok(f"反向检验样本已生成: {len(df_neg_out)} 条")
        ai_excluded = (df_neg_out['AI_Decision'] == 'AI判定整篇文章无关').sum()
        UX.info(f"其中 {ai_excluded} 条来自AI判定无关的文章")

    # =========== 3. 正向检验（精确率与边界）===========
    UX.info("生成正向检验样本...")

    # 筛选成功处理的议题单元
    df_pos_pool = df_results[df_results['processing_status'] == ProcessingStatus.SUCCESS].copy()

    positive_samples = []
    new_precision_ids = set()

    for source, cfg in RELIABILITY_SAMPLING_CONFIG.items():
        mask_source = df_pos_pool['Source'] == source
        mask_not_sampled = ~df_pos_pool['Unit_ID'].astype(str).isin(sampled_records['precision'])

        candidates = df_pos_pool[mask_source & mask_not_sampled]
        n_sample = min(int(cfg.get('precision', 0)), len(candidates))

        if n_sample > 0:
            random_seed = RANDOMIZATION_CONFIG.get('random_seed', 2025)
            sample_df = candidates.sample(n=n_sample, replace=False, random_state=random_seed)
            positive_samples.append(sample_df)
            new_precision_ids.update(sample_df['Unit_ID'].astype(str).tolist())

    if positive_samples:
        df_pos = pd.concat(positive_samples, ignore_index=True)
        df_pos = df_pos.drop_duplicates(subset=['Unit_ID'], keep='first')

        # 创建宏观块索引
        df_macro_dict = df_macro.set_index('Macro_Chunk_ID').to_dict('index')

        # 生成高亮文本
        highlighted_texts = []

        for _, row in df_pos.iterrows():
            macro_id = str(row.get('Macro_Chunk_ID', ''))
            unit_text = str(row.get('Unit_Text', ''))

            if macro_id in df_macro_dict:
                parent_text = str(df_macro_dict[macro_id].get('Macro_Chunk_Text', ''))
                highlighted = _highlight_Unit_in_parent(parent_text, unit_text)
            else:
                highlighted = f"[未找到宏观块]\n议题单元：【{unit_text}】"

            highlighted_texts.append(highlighted)

        # 构建输出数据
        df_pos_out = pd.DataFrame()
        df_pos_out['Unit_ID'] = df_pos['Unit_ID']
        df_pos_out['Source'] = df_pos['Source']
        df_pos_out['Macro_Chunk_ID'] = df_pos['Macro_Chunk_ID']
        df_pos_out['Parent_Macro_Chunk_Text_Highlighted'] = highlighted_texts
        df_pos_out['Unit_Text'] = df_pos['Unit_Text']

        # 检验员字段
        df_pos_out['Inspector_Is_CN_RU_Related'] = ''  # 是/否
        df_pos_out['Inspector_Boundary'] = ''           # 合适/偏小/偏大
        df_pos_out['Inspector_Comments'] = ''           # 备注

        # 保存文件
        zh_pos = os.path.join(output_path, '正向检验_精确率与边界样本.xlsx')
        ru_pos = os.path.join(output_path, 'Проверка_положительная_выборка(Precision_и_Границы).xlsx')
        _save_bilingual(df_pos_out, zh_pos, ru_pos)

        UX.ok(f"正向检验样本已生成: {len(df_pos_out)} 条")

        # =========== 4. 框架维度检验 ===========
        UX.info("生成框架维度检验样本...")

        # 框架字段映射（更新为新的对象格式）
        frame_fields = [
            ('ProblemDefinition', 'Frame_ProblemDefinition'),
            ('ResponsibilityAttribution', 'Frame_ResponsibilityAttribution'),
            ('MoralEvaluation', 'Frame_MoralEvaluation'),
            ('SolutionRecommendation', 'Frame_SolutionRecommendation'),
            ('ActionStatement', 'Frame_ActionStatement'),
            ('CausalExplanation', 'Frame_CausalExplanation')
        ]

        # 维度字段
        dimension_fields = [
            'Valence', 'Evidence_Type', 'Attribution_Level', 'Temporal_Focus',
            'Primary_Actor_Type', 'Geographic_Scope', 'Relationship_Model_Definition', 'Discourse_Type'
        ]

        # 构建框架检验数据
        df_frame = pd.DataFrame()
        df_frame['Unit_ID'] = df_pos['Unit_ID']
        df_frame['Source'] = df_pos['Source']
        df_frame['Macro_Chunk_ID'] = df_pos['Macro_Chunk_ID']
        df_frame['Parent_Macro_Chunk_Text_Highlighted'] = highlighted_texts
        df_frame['Unit_Text'] = df_pos['Unit_Text']

        # 处理框架字段
        for display_name, field_name in frame_fields:
            if field_name in df_pos.columns:
                # AI识别结果
                ai_values = []
                ai_quotes = []
                inspector_col = f'Inspector_Frame_{display_name}_Present'

                for _, row in df_pos.iterrows():
                    val = row.get(field_name, [])

                    # 处理新的对象格式 [{"quote": "...", "reason": "...", "reasoning_pattern": "..."}]
                    if isinstance(val, list):
                        has_frame = len(val) > 0
                        if val and isinstance(val[0], dict):
                            # 新的对象格式：提取所有quote字段
                            quotes = [item.get('quote', '') for item in val if isinstance(item, dict) and item.get('quote')]
                            quote_text = '; '.join(quotes) if quotes else ''
                        else:
                            # 兼容旧格式：直接使用字符串列表
                            quote_text = '; '.join([str(q) for q in val]) if val else ''
                    elif isinstance(val, str):
                        val_str = str(val).strip()
                        has_frame = len(val_str) > 2 and val_str != '[]'
                        quote_text = val_str if has_frame else ''
                    else:
                        has_frame = False
                        quote_text = ''

                    ai_values.append(1 if has_frame else 0)
                    ai_quotes.append(quote_text)

                # 添加到数据框
                df_frame[f'AI_Frame_{display_name}_Present'] = ai_values
                df_frame[f'AI_Frame_{display_name}_Quotes'] = ai_quotes  # 添加引文内容
                df_frame[inspector_col] = ''  # 检验员判断

        # 处理维度字段
        for dim in dimension_fields:
            if dim in df_pos.columns:
                df_frame[f'AI_{dim}'] = df_pos[dim]  # AI的分类结果
                df_frame[f'Inspector_{dim}_Correct'] = ''  # 检验员判断是否正确

        # 保存框架维度检验文件
        zh_frame = os.path.join(output_path, '框架维度检验_单检验员.xlsx')
        ru_frame = os.path.join(output_path, 'Проверка_Рамки_и_Размерности_одним_проверяющим.xlsx')
        _save_bilingual(df_frame, zh_frame, ru_frame)

        UX.ok(f"框架维度检验样本已生成: {len(df_frame)} 条")

    # =========== 5. 保存抽样记录 ===========
    if new_recall_ids or new_precision_ids:
        sampled_records['recall'].update(new_recall_ids)
        sampled_records['precision'].update(new_precision_ids)

        try:
            with open(sampled_cache, 'wb') as f:
                pickle.dump(sampled_records, f)
            UX.info(f"更新抽样记录: 累计召回{len(sampled_records['recall'])}条, "
                   f"精确{len(sampled_records['precision'])}条")
        except Exception as e:
            UX.warn(f"保存抽样记录失败: {e}")

    UX.ok("所有信度检验文件生成完成！")

    # 输出统计信息
    print("\n" + "="*60)
    print("信度检验文件生成统计：")
    print("-"*60)

    if negative_samples:
        print(f"反向检验（召回率）: {len(df_neg_out)} 条样本")
        print(f"  - AI判定无关: {(df_neg_out['AI_Decision']=='AI判定整篇文章无关').sum()} 条")
        print(f"  - 未被使用: {(df_neg_out['AI_Decision']=='未被使用的宏观块').sum()} 条")

    if positive_samples:
        print(f"正向检验（精确率）: {len(df_pos_out)} 条样本")
        print(f"框架维度检验: {len(df_frame)} 条样本")

    print("="*60)

# ============================================================================
# 信度检验验证和诊断工具
# ============================================================================

def verify_macro_chunks_quality(macro_chunks):
    """验证宏观块数据质量"""
    if not macro_chunks:
        return False, "没有宏观块数据"

    issues = []

    # 检查必要字段
    sample = macro_chunks[0]
    required = ['Macro_Chunk_ID', 'Macro_Chunk_Text', 'Speaker', 'Source']
    missing = [f for f in required if f not in sample]

    if missing:
        issues.append(f"缺少字段: {missing}")

    # 检查文本长度（字符数和token数）
    char_lengths = [len(str(m.get('Macro_Chunk_Text', ''))) for m in macro_chunks]
    token_lengths = [count_tokens(str(m.get('Macro_Chunk_Text', ''))) for m in macro_chunks]

    avg_char_len = sum(char_lengths) / len(char_lengths) if char_lengths else 0
    avg_token_len = sum(token_lengths) / len(token_lengths) if token_lengths else 0

    min_chars_strict = QUALITY_THRESHOLDS.get('min_macro_chunk_chars_strict', 100)
    if avg_char_len < min_chars_strict:
        issues.append(f"宏观块平均长度过短: {avg_char_len:.0f} 字符 (< {min_chars_strict})")

    # 检查是否有空文本
    empty_count = sum(1 for m in macro_chunks if not str(m.get('Macro_Chunk_Text', '')).strip())
    if empty_count > 0:
        issues.append(f"有 {empty_count} 个空宏观块")

    if issues:
        return False, "; ".join(issues)

    return True, f"通过验证: {len(macro_chunks)} 个宏观块，平均长度 {avg_char_len:.0f} 字符, {avg_token_len:.0f} tokens"

if __name__ == "__main__":
    asyncio.run(main_async())