# -*- coding: utf-8 -*-
"""
Utility functions and classes for the media and social media text analysis system.
"""

import os
import re
import time
import yaml
import tiktoken
import pandas as pd
import contextlib
from datetime import datetime

# ==============================================================================
# 处理状态常量类 - 基础组件，不依赖其他模块
# ==============================================================================

class ProcessingStatus:
    """处理状态常量"""
    SUCCESS = "SUCCESS"
    NO_RELEVANT = "NO_RELEVANT"
    API_FAILED = "API_FAILED"

# 分析列常量 - 从processors.py移至此处
ANALYSIS_COLUMNS = [
    'Incident', 'Frame_SolutionRecommendation', 'Frame_ResponsibilityAttribution',
    'Frame_CausalExplanation', 'Frame_MoralEvaluation', 'Frame_ProblemDefinition',
    'Frame_ActionStatement', 'Valence', 'Evidence_Type', 'Attribution_Level',
    'Temporal_Focus', 'Primary_Actor_Type', 'Geographic_Scope',
    'Relationship_Model_Definition', 'Discourse_Type'
]

class UX:
    @staticmethod
    def _fmt(msg):
        return str(msg).strip()
    
    # 运行级计时起点
    RUN_T0 = time.perf_counter()
    LAST_LOG_TIME = time.perf_counter()  # 最后一次日志时间
    
    @staticmethod
    def start_run():
        UX.RUN_T0 = time.perf_counter()
        UX.LAST_LOG_TIME = time.perf_counter()

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
    def _update_log_time():
        """更新最后日志时间"""
        UX.LAST_LOG_TIME = time.perf_counter()
    
    @staticmethod
    def check_activity():
        """检查是否需要输出活动提醒"""
        if time.perf_counter() - UX.LAST_LOG_TIME > 300:  # 5分钟 = 300秒
            print(f"[{UX._ts()}][{UX._elapsed_str()}] [⏳] 算法仍在运行中...")
            UX.LAST_LOG_TIME = time.perf_counter()
    
    @staticmethod
    def phase(title):
        print(f"\n=== [{UX._ts()}][{UX._elapsed_str()}] {UX._fmt(title)} ===")
        UX._update_log_time()
    
    @staticmethod
    def resume_plan(title):
        """断点续传专用格式，更显眼"""
        print(f"\n" + "="*100)
        print("="*100)
        print(f"=== 📋 断点续传计划 - {UX._fmt(title)} ===")
        print("="*100)
        UX._update_log_time()
    
    @staticmethod
    def resume_end():
        """断点续传结束"""
        print("="*100)
        print("="*100 + "\n")
        UX._update_log_time()
    
    @staticmethod
    def info(msg):
        print(f"[{UX._ts()}][{UX._elapsed_str()}] [i] {UX._fmt(msg)}")
        UX._update_log_time()
    
    @staticmethod
    def ok(msg):
        print(f"[{UX._ts()}][{UX._elapsed_str()}] [OK] {UX._fmt(msg)}")
        UX._update_log_time()
    
    @staticmethod
    def warn(msg):
        print(f"[{UX._ts()}][{UX._elapsed_str()}] [!] {UX._fmt(msg)}")
        UX._update_log_time()
    
    @staticmethod
    def err(msg):
        print(f"[{UX._ts()}][{UX._elapsed_str()}] [!!!] {UX._fmt(msg)}")
        UX._update_log_time()
    
    @staticmethod
    def api_failed(stage, error_brief):
        """API失败简要日志"""
        print(f"[{UX._ts()}][{UX._elapsed_str()}] [❌] {stage}: {error_brief}")
        UX._update_log_time()
    
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


def load_config():
    """加载配置文件"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
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
        return len(text) // 2  # 粗略估算：1 token ≈ 2 字符


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
        return 'zh'
    elif re.search(r'[\u0400-\u04FF]', text):
        return 'ru'
    else:
        return 'en'


def identify_source(filename):
    source_map = {
        '俄总统': ['俄总统', '总统', 'Putin', 'president'],
        '俄语媒体': ['俄语媒体', '俄语', 'russian', 'ru_media', '俄媒'],
        '中文媒体': ['中文媒体', '中文', 'chinese', 'cn_media', '中媒'],
        '英语媒体': ['英语媒体', '英语', 'english', 'en_media', '英媒'],
        'vk': ['vk'],
        '知乎': ['知乎', 'zhihu']
    }

    filename_lower = filename.lower()
    for source, keywords in source_map.items():
        if any(kw.lower() in filename_lower for kw in keywords):
            UX.info(f"文件 {filename} 识别为: {source}")
            return source

    UX.warn(f"文件 {filename} 无法识别信源，标记为: 未知来源")
    return '未知来源'

# ==============================================================================
# 数据处理辅助函数 - 从 processors.py 移至此处
# ==============================================================================

# get_language_config函数已删除，请从config.yaml获取语言配置

def get_processing_state(df, id_col):
    """统一的状态检查：返回(完全成功ID集合, 有失败的ID集合)"""
    # ProcessingStatus现在在本文件中定义
    
    if df is None or df.empty or id_col not in df.columns:
        return set(), set()
    
    status_col = 'processing_status'
    try:
        if status_col in df.columns:
            # 基于文章维度判断状态
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

def clean_failed_records(output_path, id_column):
    """清理失败记录"""
    # ProcessingStatus现在在本文件中定义
    
    if not os.path.exists(output_path):
        return set()

    try:
        df = pd.read_excel(output_path)
        if df.empty or id_column not in df.columns:
            return set()

        if 'processing_status' in df.columns:
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

def create_unified_record(record_type, original_id, source="未知来源", text_snippet="", failure_reason=""):
    """创建统一记录"""
    # ProcessingStatus现在在本文件中定义
    
    base_record = {
        "processing_status": record_type,
        "Source": source,
        "Unit_ID": f"{original_id}-{record_type}"
    }

    if record_type == ProcessingStatus.NO_RELEVANT:
        return {
            **base_record,
            "speaker": "NO_RELEVANT_CONTENT",
            "Unit_Text": "[无相关内容]",
            "Incident": "",
            "Frame_SolutionRecommendation": "",
            "Frame_ResponsibilityAttribution": "",
            "Frame_CausalExplanation": "",
            "Frame_MoralEvaluation": "",
            "Frame_ProblemDefinition": "",
            "Frame_ActionStatement": "",
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
            "speaker": "API_CALL_FAILED",
            "Unit_Text": f"[API_FAILED] {failure_reason}: {text_snippet[:200]}...",
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

def detect_language_and_get_config(text, config=None):
    """检测语言并获取配置"""
    import re
    
    # 默认配置
    default_configs = {
        'zh': {'MAX_SINGLE_TEXT': 50000},
        'ru': {'MAX_SINGLE_TEXT': 50000}, 
        'en': {'MAX_SINGLE_TEXT': 50000}
    }
    
    # 使用提供的配置或默认配置
    language_configs = config.get('LANGUAGE_CONFIGS', default_configs) if config else default_configs
    
    if re.search(r'[\u4e00-\u9fa5]', text):
        return 'zh', language_configs.get('zh', default_configs['zh'])
    elif re.search(r'[\u0400-\u04FF]', text):
        return 'ru', language_configs.get('ru', default_configs['ru'])
    else:
        return 'en', language_configs.get('en', default_configs['en'])
