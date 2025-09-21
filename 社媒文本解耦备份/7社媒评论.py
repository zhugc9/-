# -*- coding: utf-8 -*-
"""
==============================================================================
社交媒体议题单元分析器 - 与媒体文本功能对齐版
==============================================================================
"""
import os
import re
import json
import time
import pandas as pd
import asyncio
import aiohttp
import hashlib
import yaml
import tiktoken
from datetime import datetime
from tqdm.asyncio import tqdm as aio_tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import contextlib

# 获取脚本所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 加载外部配置文件
def load_config():
    """加载YAML配置文件"""
    yaml_path = os.path.join(BASE_DIR, "config.yaml")
    
    if not os.path.exists(yaml_path):
        print(f"❌ YAML配置文件不存在: {yaml_path}")
        print("请确保config.yaml文件存在并包含所有必需的参数。")
        raise FileNotFoundError(f"YAML配置文件不存在: {yaml_path}")
    
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            print(f"✅ 成功加载YAML配置文件: {yaml_path}")
            return config
    except yaml.YAMLError as e:
        print(f"❌ YAML配置文件格式错误: {e}")
        print("请检查YAML格式是否正确。")
        raise ValueError(f"YAML配置文件格式错误: {e}")
    except Exception as e:
        print(f"❌ 加载YAML配置文件失败: {e}")
        raise FileNotFoundError(f"加载YAML配置文件失败: {e}")

# 全局配置
CONFIG = load_config()

class UX:
    """统一的用户交互管理器"""
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
        hours, rem = divmod(total, 3600)
        minutes, seconds = divmod(rem, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    @staticmethod
    def phase(title):
        print(f"\n=== [{UX._ts()}][{UX._elapsed_str()}] {title} ===")

    @staticmethod
    def info(msg):
        print(f"[{UX._ts()}][{UX._elapsed_str()}] [i] {msg}")

    @staticmethod
    def ok(msg):
        print(f"[{UX._ts()}][{UX._elapsed_str()}] [OK] {msg}")

    @staticmethod
    def warn(msg):
        print(f"[{UX._ts()}][{UX._elapsed_str()}] [!] {msg}")

    @staticmethod
    def err(msg):
        print(f"[{UX._ts()}][{UX._elapsed_str()}] [!!!] {msg}")

# ==============================================================================
# === 🎛️ 核心参数配置区（已外部化）
# ==============================================================================

# 从外部配置文件加载所有参数
INPUT_PATH = CONFIG['INPUT_PATH']
OUTPUT_PATH = CONFIG['OUTPUT_PATH']
RELIABILITY_TEST_MODE = CONFIG['RELIABILITY_TEST_MODE']
RELIABILITY_SAMPLING_CONFIG = CONFIG['RELIABILITY_SAMPLING_CONFIG']
API_CONFIG = CONFIG['API_CONFIG']
MAX_CONCURRENT_REQUESTS = CONFIG['MAX_CONCURRENT_REQUESTS']
API_RETRY_ATTEMPTS = CONFIG['API_RETRY_ATTEMPTS']
RATE_LIMIT_BASE_DELAY = CONFIG['RATE_LIMIT_BASE_DELAY']
SKIP_FAILED_TEXTS = CONFIG['SKIP_FAILED_TEXTS']
VK_LONG_TEXT_THRESHOLD = CONFIG['VK_LONG_TEXT_THRESHOLD']
ZHIHU_SHORT_TOKEN_THRESHOLD = CONFIG['ZHIHU_SHORT_TOKEN_THRESHOLD']
ZHIHU_LONG_TOKEN_THRESHOLD = CONFIG['ZHIHU_LONG_TOKEN_THRESHOLD']
LANGUAGE_CONFIGS = CONFIG['LANGUAGE_CONFIGS']
COLUMN_MAPPING = CONFIG['COLUMN_MAPPING']

# 处理状态定义
class ProcessingStatus:
    SUCCESS = "SUCCESS"
    NO_RELEVANT = "NO_RELEVANT"
    API_FAILED = "API_FAILED"

# 分析列定义（V2格式 - 直接使用新的框架字段）
ANALYSIS_COLUMNS = ['Source', 'Incident', 'relevance', 'speaker',
                   'Frame_ProblemDefinition', 
                   'Frame_ResponsibilityAttribution',
                   'Frame_MoralEvaluation',
                   'Frame_SolutionRecommendation',
                   'Frame_ActionStatement',
                   'Frame_CausalExplanation',
                   'Valence', 'Evidence_Type', 'Attribution_Level',
                   'Temporal_Focus', 'Primary_Actor_Type', 'Geographic_Scope',
                   'Relationship_Model_Definition', 'Discourse_Type']

# ==============================================================================
# === 🔧 核心功能模块
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
        return len(text) // 2  # 粗略估算：1 token ≈ 2 字符

class Utils:
    """统一工具函数类"""
    
    @staticmethod
    def safe_str_convert(value):
        """安全字符串转换"""
        if pd.isna(value) or value is None:
            return ''
        return str(value)

    @staticmethod
    def normalize_text(text):
        """文本规范化"""
        if not isinstance(text, str):
            text = Utils.safe_str_convert(text)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def detect_language(text):
        """检测语言"""
        if re.search(r'[\u4e00-\u9fa5]', text):
            return 'zh'
        elif re.search(r'[\u0400-\u04FF]', text):
            return 'ru'
        return 'en'

    @staticmethod
    def get_language_config(language):
        """获取语言对应的处理配置"""
        return LANGUAGE_CONFIGS.get(language, LANGUAGE_CONFIGS['en'])

    @staticmethod
    def detect_file_type(df):
        """检测文件类型（VK或知乎）"""
        vk_columns = set(COLUMN_MAPPING['vk'].values())
        zhihu_required = {'序号', '知乎问题标题及描述', '回答内容'}
        df_columns = set(df.columns)
        
        if vk_columns.issubset(df_columns):
            return 'vk'
        elif zhihu_required.issubset(df_columns):
            return 'zhihu'
        return None

    @staticmethod
    def identify_source(filename):
        """识别信源"""
        filename_lower = filename.lower()
        if 'vk' in filename_lower:
            return 'vk'
        elif '知乎' in filename or 'zhihu' in filename_lower:
            return '知乎'
        return '未知来源'
    
    @staticmethod
    def safe_json_parse(json_str, default=None):
        """安全解析JSON字符串"""
        if default is None:
            default = []
        if pd.isna(json_str) or json_str is None or json_str == '':
            return default
        if isinstance(json_str, list):
            return json_str
        if isinstance(json_str, str):
            try:
                return json.loads(json_str)
            except (json.JSONDecodeError, TypeError):
                return default
        return default
    
    @staticmethod
    def safe_json_dumps(obj, ensure_ascii=True):
        """安全序列化JSON对象"""
        if obj is None or (isinstance(obj, list) and len(obj) == 0):
            return '[]'
        try:
            return json.dumps(obj, ensure_ascii=ensure_ascii)
        except (TypeError, ValueError):
            return '[]'
    

# 保持向后兼容性的别名
safe_str_convert = Utils.safe_str_convert
normalize_text = Utils.normalize_text
detect_language = Utils.detect_language
get_language_config = Utils.get_language_config
detect_file_type = Utils.detect_file_type
identify_source = Utils.identify_source

def create_unified_record(record_type, id_value, source="未知来源", text_snippet="", failure_reason=""):
    """创建统一格式记录（与媒体文本一致）"""
    base = {
        "processing_status": record_type,
        "Source": source,
        "Unit_ID": f"{id_value}-{record_type}"
    }
    
    # 框架和维度字段（与Prompts类保持一致）
    frames = ["ProblemDefinition", "ResponsibilityAttribution", "MoralEvaluation",
              "SolutionRecommendation", "ActionStatement", "CausalExplanation"]
    dims = ["Valence", "Evidence_Type", "Attribution_Level", "Temporal_Focus",
            "Primary_Actor_Type", "Geographic_Scope", "Relationship_Model_Definition", 
            "Discourse_Type"]
    
    if record_type == ProcessingStatus.API_FAILED:
        base.update({
            "Unit_Text": f"[API_FAILED] {failure_reason}: {text_snippet[:200]}...",
            "speaker": "API_CALL_FAILED",
            "Incident": "API_CALL_FAILED",
            "relevance": "API_FAILED",  # VK特有字段
            **{f"Frame_{f}": [] for f in frames},
            **{d: "API_CALL_FAILED" for d in dims}
        })
    elif record_type == ProcessingStatus.NO_RELEVANT:
        base.update({
            "Unit_Text": "[无相关内容]",
            "speaker": "NO_RELEVANT_CONTENT",
            "Incident": "NO_RELEVANT_CONTENT",
            "relevance": "不相关",  # VK特有字段
            **{f"Frame_{f}": [] for f in frames},
            **{d: "NO_RELEVANT_CONTENT" for d in dims}
        })
    
    return base

def clean_failed_records(output_path, id_column):
    """清理失败记录（与媒体文本对齐）"""
    if not os.path.exists(output_path):
        return set()
    
    try:
        df = pd.read_excel(output_path)
        if df.empty:
            return set()
        
        if 'processing_status' in df.columns:
            # 只清理API_FAILED的记录，保留NO_RELEVANT记录
            failed_mask = df['processing_status'] == ProcessingStatus.API_FAILED
        else:
            # 兼容旧版本
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

def get_processing_state(df, id_col):
    """统一的状态检查：返回(成功ID集合, 失败ID集合)（与媒体文本一致）"""
    if df is None or df.empty or id_col not in df.columns:
        return set(), set()
    
    status_col = 'processing_status'
    try:
        if status_col in df.columns:
            # SUCCESS和NO_RELEVANT都算成功，只有API_FAILED需要重新处理
            success = df[df[status_col].isin([ProcessingStatus.SUCCESS, ProcessingStatus.NO_RELEVANT])][id_col]
            failed = df[df[status_col] == ProcessingStatus.API_FAILED][id_col]
        else:
            # 兼容旧版
            speaker_col = 'speaker'
            if speaker_col in df.columns:
                success = df[~df[speaker_col].astype(str).str.contains('API_CALL_FAILED', na=False)][id_col]
                failed = df[df[speaker_col].astype(str).str.contains('API_CALL_FAILED', na=False)][id_col]
            else:
                return set(), set()
        
        return set(success.astype(str)), set(failed.astype(str))
    except Exception:
        return set(), set()

def get_failed_batch_ids(post_id, output_file_path):
    """获取指定帖子中失败的批处理ID（类似媒体文本的get_failed_macro_chunk_ids）"""
    if not os.path.exists(output_file_path):
        return set()
    
    try:
        df_output = pd.read_excel(output_file_path)
        if df_output.empty:
            return set()
        
        # 筛选指定帖子的记录
        if 'Post_ID' in df_output.columns:
            post_records = df_output[df_output['Post_ID'].astype(str) == str(post_id)]
        else:
            return set()
        
        if post_records.empty:
            return set()
        
        # 获取失败的记录
        failed_mask = post_records['processing_status'] == ProcessingStatus.API_FAILED
        failed_records = post_records[failed_mask]
        
        if failed_records.empty:
            return set()
        
        # 获取批处理ID（如果有的话）
        failed_batch_ids = set()
        if 'Batch_ID' in failed_records.columns:
            failed_batch_ids = set(failed_records['Batch_ID'].dropna().astype(str))
        
        return failed_batch_ids
    except Exception as e:
        UX.warn(f"获取失败批处理ID失败: {e}")
        return set()

# ==============================================================================
# === 🤖 API服务
# ==============================================================================

class APIService:
    """统一的API服务"""
    
    def __init__(self, session=None):
        self.session = session
        self.call_count = 0
        self.success_count = 0
        self.failure_count = 0
    
    def _create_payload(self, prompt, stage_key):
        """创建API请求负载"""
        return {
            "model": API_CONFIG["STAGE_MODELS"].get(stage_key, "[官自-0.7]gemini-2-5-flash"),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "response_format": {"type": "json_object"}
        }
    
    def _extract_json_response(self, content):
        """从响应中提取JSON（支持对象和数组）"""
        # 尝试找到JSON数组
        first_bracket = content.find('[')
        last_bracket = content.rfind(']')
        
        # 尝试找到JSON对象
        first_brace = content.find('{')
        last_brace = content.rfind('}')
        
        # 判断哪个在前（优先处理最外层的结构）
        if first_bracket >= 0 and (first_brace < 0 or first_bracket < first_brace):
            # 是数组
            if last_bracket > first_bracket:
                try:
                    return json.loads(content[first_bracket:last_bracket+1])
                except json.JSONDecodeError:
                    pass
        
        if first_brace >= 0 and last_brace > first_brace:
            # 是对象
            try:
                return json.loads(content[first_brace:last_brace+1])
            except json.JSONDecodeError:
                pass
        
        # 最后尝试直接解析整个内容
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            raise ValueError(f"无法解析JSON: {content[:200]}...")
    
    def print_statistics(self):
        """打印API调用统计"""
        if self.call_count > 0:
            success_rate = (self.success_count / self.call_count) * 100
            UX.info(f"API统计 - 总调用: {self.call_count}, 成功: {self.success_count}, "
                    f"失败: {self.failure_count}, 成功率: {success_rate:.1f}%")
    
    async def call_api_async(self, prompt, language='zh', stage_key=None):
        """异步API调用"""
        self.call_count += 1
        config = LANGUAGE_CONFIGS.get(language, LANGUAGE_CONFIGS['zh'])
        
        for attempt in range(API_RETRY_ATTEMPTS):
            try:
                url = f"{API_CONFIG['BASE_URL']}/chat/completions"
                headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_CONFIG['API_KEYS'][0]}"}
                payload = self._create_payload(prompt, stage_key)
                
                
                async with self.session.post(url, headers=headers, json=payload,
                                            timeout=aiohttp.ClientTimeout(total=config['TIMEOUT'])) as response:
                    response.raise_for_status()
                    result = json.loads(await response.text())
                    content = result['choices'][0]['message']['content']
                    
                    result = self._extract_json_response(content)
                    self.success_count += 1
                    return result
                    
            except Exception as e:
                if attempt < API_RETRY_ATTEMPTS - 1:
                    await asyncio.sleep(RATE_LIMIT_BASE_DELAY * (2 ** attempt))
                else:
                    self.failure_count += 1
                    if SKIP_FAILED_TEXTS:
                        UX.warn(f"API失败: {str(e)[:100]}")
                        return None
                    raise
        return None

    def call_api_sync(self, prompt, language='zh', stage_key=None):
        """同步API调用"""
        import requests
        self.call_count += 1
        config = LANGUAGE_CONFIGS.get(language, LANGUAGE_CONFIGS['zh'])
        
        for attempt in range(API_RETRY_ATTEMPTS):
            try:
                url = f"{API_CONFIG['BASE_URL']}/chat/completions"
                headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_CONFIG['API_KEYS'][0]}"}
                payload = self._create_payload(prompt, stage_key)
                
                
                response = requests.post(url, headers=headers, json=payload, timeout=config['TIMEOUT'])
                response.raise_for_status()
                content = response.json()['choices'][0]['message']['content']
                
                result = self._extract_json_response(content)
                self.success_count += 1
                return result
                
            except Exception as e:
                if attempt < API_RETRY_ATTEMPTS - 1:
                    time.sleep(RATE_LIMIT_BASE_DELAY * (2 ** attempt))
                else:
                    self.failure_count += 1
                    if SKIP_FAILED_TEXTS:
                        UX.warn(f"API失败: {str(e)[:100]}")
                        return None
                    raise
        return None

# ==============================================================================
# === 🤖 提示词模板
# ==============================================================================

class Prompts:
    """提示词管理类 - 从外部文件加载"""
    
    def __init__(self):
        self.prompts_dir = BASE_DIR
        self._load_prompts()
    
    def _load_prompts(self):
        """从外部文件加载所有提示词"""
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
# ==============================================================================
# === ⚙️ 处理器
# ==============================================================================

class BaseProcessor:
    """基础处理器类"""
    
    def __init__(self, api_service):
        self.api_service = api_service
        self.Units_collector = []
    
    def _add_hash_to_record(self, record, text_field='Unit_Text'):
        """为记录添加哈希值"""
        if text_field in record:
            norm = normalize_text(record[text_field])
            if norm:
                record['Unit_Hash'] = hashlib.sha256(norm.encode('utf-8')).hexdigest()
        return record
    
    def _save_progress_generic(self, df_to_process, output_path, id_column):
        """通用保存进度方法（与媒体文本对齐）"""
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
        
        mapping = COLUMN_MAPPING['vk']
        
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
        
        # 【关键修复】：初始化为PENDING而不是SUCCESS
        df_to_process['processing_status'] = 'PENDING'  # 待处理状态
        
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
        
        config = get_language_config('ru')
        
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
            for i in range(0, len(comments_list), config['BATCH_SIZE_LIMIT']):
                chunk = comments_list[i:i + config['BATCH_SIZE_LIMIT']]
                chunk_ids = comment_ids_in_batch[i:i + config['BATCH_SIZE_LIMIT']]
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
        with ThreadPoolExecutor(max_workers=config['MAX_CONCURRENT']) as executor:
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
                                    df_to_process.loc[mask, key] = Utils.safe_json_dumps(value, ensure_ascii=False)
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
                    if (batch_idx + 1) % config['SAVE_INTERVAL'] == 0:
                        self._save_progress_generic(df_to_process, output_path, mapping['comment_id'])
                        # 显示当前统计
                        success_count = (df_to_process['processing_status'] == ProcessingStatus.SUCCESS).sum()
                        failed_count = (df_to_process['processing_status'] == ProcessingStatus.API_FAILED).sum()
                        no_relevant_count = (df_to_process['processing_status'] == ProcessingStatus.NO_RELEVANT).sum()
                        pending_count = (df_to_process['processing_status'] == 'PENDING').sum()
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
        pending_mask = df_to_process['processing_status'] == 'PENDING'
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
        """处理单个批次 - 修复版"""
        try:
            post_text = batch_task['post_text']
            comments_json = json.dumps(batch_task['comments'], ensure_ascii=False)
            
            prompt = prompts.VK_BATCH_ANALYSIS.format(
                post_text=str(post_text),
                comments_json=comments_json
            )
            
            # 智能模型选择：根据文本长度选择合适的模型
            combined_text = str(post_text) + " " + comments_json
            text_tokens = count_tokens(combined_text)
            
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
                # 返回失败记录列表，添加批处理标记
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
                        
                        # V2格式直接使用，无需转换
                        
                        self._add_hash_to_record(item, 'comment_text')  # 直接基于comment_text生成哈希
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
                                
                                # V2格式直接使用，无需转换
                                
                                self._add_hash_to_record(item, 'comment_text')  # 直接基于comment_text生成哈希
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
                # 为缺失的评论添加失败记录
                post_id = batch_task.get('post_id')
                for comment_id in missing_ids:
                    processed_results.append(self._create_failed_record(comment_id, 'API响应中缺失', post_id))
            
            if extra_ids:
                UX.warn(f"API响应包含 {len(extra_ids)} 条额外评论，将被忽略")
                # 过滤掉额外的结果
                processed_results = [r for r in processed_results if str(r.get('comment_id', '')) in expected_ids]
                        
            return processed_results
            
        except Exception as e:
            UX.err(f"批次处理异常: {str(e)}")
            # 返回异常记录，添加批处理标记
            post_id = batch_task.get('post_id')
            return [self._create_failed_record(c['comment_id'], f'异常: {str(e)[:50]}', post_id) 
                    for c in batch_task['comments']]
    
    def _create_failed_record(self, comment_id, reason, post_id=None):
        """创建失败记录（复用统一函数）"""
        record = create_unified_record(ProcessingStatus.API_FAILED, comment_id, 'vk', '', reason)
        # 添加VK特有的字段
        record['comment_id'] = comment_id
        record['relevance'] = 'API_FAILED'
        record['speaker'] = 'API_CALL_FAILED'  # 保持一致性
        record['Incident'] = reason
        
        # 添加批处理特殊标记（类似媒体文本的Macro_Chunk_ID）
        if post_id:
            record['Batch_ID'] = f"{post_id}-BATCH_FAILED"  # 批处理失败标记
        
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
        
        mapping = COLUMN_MAPPING['zhihu']
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
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        tasks = []
        
        for idx, row in df_to_process.iterrows():
            answer_id = str(row[mapping["id"]])
            tasks.append(self._process_answer(row, mapping, answer_id, semaphore))
        
        all_results = await aio_tqdm.gather(*tasks)
        
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
                if answer_tokens < ZHIHU_SHORT_TOKEN_THRESHOLD:
                    # === 短文本模式：直接分析（类似VK） ===
                    UX.info(f"🔧 知乎回答 {answer_id} 模式选择: ({answer_tokens} tokens) → 短文本直接分析模式")
                    
                    # 使用模板构造提示词
                    prompt = prompts.ZHIHU_SHORT_ANALYSIS.format(
                        question=question,
                        answer_text=answer_text
                    )
                    
                    # 智能模型选择：短文本模式直接使用轻量模型
                    stage_key = 'ZHIHU_ANALYSIS_SHORT'
                    UX.info(f"🔧 知乎模型选择: ({answer_tokens} tokens < {ZHIHU_SHORT_TOKEN_THRESHOLD}) → 轻量模型")
                    
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
                    prompt1 = prompts.ZHIHU_CHUNKING.format(full_text=answer_text)
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
                        
                        prompt2 = prompts.ZHIHU_ANALYSIS.format(
                            question=question,
                            Unit_Text=Unit_Text
                        )
                        
                        # 智能模型选择：根据议题单元长度选择合适的模型
                        unit_tokens = count_tokens(Unit_Text)
                        if unit_tokens < ZHIHU_SHORT_TOKEN_THRESHOLD:
                            stage_key = 'ZHIHU_ANALYSIS_SHORT'
                            UX.info(f"🔧 知乎议题单元模型选择: ({unit_tokens} tokens < {ZHIHU_SHORT_TOKEN_THRESHOLD}) → 轻量模型")
                        elif unit_tokens > ZHIHU_LONG_TOKEN_THRESHOLD:
                            stage_key = 'ZHIHU_ANALYSIS_LONG'
                            UX.info(f"🔧 知乎议题单元模型选择: ({unit_tokens} tokens > {ZHIHU_LONG_TOKEN_THRESHOLD}) → 高性能模型")
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

# ==============================================================================
# === 🔬 信度检验（与媒体文本对齐）
# ==============================================================================

def save_Units_database(Units_data, output_path):
    """保存基础单元数据库"""
    if not Units_data:
        return None
    
    database_path = os.path.join(output_path, '社交媒体_基础单元数据库.xlsx')
    
    if os.path.exists(database_path):
        df_existing = pd.read_excel(database_path)
        existing_ids = set(df_existing['Unit_ID'].unique())
        new_data = [u for u in Units_data if u['Unit_ID'] not in existing_ids]
        
        if new_data:
            df_new = pd.DataFrame(new_data)
            df_final = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_final = df_existing
    else:
        df_final = pd.DataFrame(Units_data)
    
    df_final.to_excel(database_path, index=False)
    UX.ok(f"基础单元数据库已保存: {len(df_final)} 条记录")
    
    return database_path

def save_parent_texts_database(parent_texts_data, output_path):
    """保存父文本数据库（帖子/回答）"""
    if not parent_texts_data:
        return None
    
    database_path = os.path.join(output_path, '社交媒体_父文本数据库.xlsx')
    
    if os.path.exists(database_path):
        df_existing = pd.read_excel(database_path)
        # 根据数据源确定ID字段
        id_col = 'Post_ID' if 'Post_ID' in df_existing.columns else 'Answer_ID'
        existing_ids = set(df_existing[id_col].unique())
        new_data = [m for m in parent_texts_data if m[id_col] not in existing_ids]
        
        if new_data:
            df_new = pd.DataFrame(new_data)
            df_final = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_final = df_existing
    else:
        df_final = pd.DataFrame(parent_texts_data)
    
    df_final.to_excel(database_path, index=False)
    UX.ok(f"父文本数据库已保存: {len(df_final)} 条记录")
    
    return database_path

# ==============================================================================
# === 🌐 双语支持功能（与媒体文本对齐）
# ==============================================================================

def _highlight_Unit_in_parent(parent_text: str, Unit_Text: str) -> tuple:
    """智能高亮：使用编辑距离找最佳匹配位置"""
    import difflib
    try:
        if not parent_text or not Unit_Text:
            return parent_text, False
        
        def normalize_spaces(s):
            return ' '.join(s.split())
        
        parent_norm = normalize_spaces(safe_str_convert(parent_text))
        Unit_norm = normalize_spaces(safe_str_convert(Unit_Text))
        
        # 1) 直接查找
        if Unit_norm in parent_norm:
            idx = parent_norm.find(Unit_norm)
            return parent_norm[:idx] + "【" + Unit_norm + "】" + parent_norm[idx+len(Unit_norm):], True
        
        # 2) 使用序列匹配器窗口搜索
        Unit_len = len(Unit_norm)
        if Unit_len == 0:
            return parent_text, False
        matcher = difflib.SequenceMatcher(None, '', Unit_norm)
        best_ratio = 0.0
        best_start = 0
        window_size = max(1, int(Unit_len * 1.2))
        min_start = 0
        max_start = max(0, len(parent_norm) - max(1, int(Unit_len * 0.8)) + 1)
        for i in range(min_start, max_start):
            end = min(i + window_size, len(parent_norm))
            window = parent_norm[i:end]
            matcher.set_seq1(window)
            ratio = matcher.ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_start = i
        
        if best_ratio > 0.85:
            end = min(best_start + window_size, len(parent_norm))
            return parent_norm[:best_start] + "【" + parent_norm[best_start:end] + "】" + parent_norm[end:], True
        
        return parent_text, False
    except Exception as e:
        UX.warn(f"高亮处理失败: {e}")
        return safe_str_convert(parent_text), False

class BilingualSupport:
    """双语支持功能 - 简化版"""
    
    # 基础映射模板
    _BASE_MAPPINGS = {
        'zh': {
            # 基础字段
            'Unit_ID': '议题单元标识符', 'Source': '数据来源', 'Post_ID': '帖子标识符', 'Comment_ID': '评论标识符',
            'Answer_ID': '回答标识符', 'Unit_Text': '议题单元文本', 'Unit_Hash': '文本哈希值',
            'speaker': '发言人', 'Incident': '核心事件概括', 'expansion_logic': '分块逻辑说明',
            'processing_status': '处理状态', 'comment_text': '评论文本', 'Post_Text': '帖子文本', 'Comment_Text': '评论文本',
            'Question': '知乎问题', 'Answer_Text': '知乎回答文本', 'Author': '作者',
            'Original_Post_ID': '原始帖子ID', 'Original_Comment_ID': '原始评论ID',
            # 分析维度
            'Valence': '情感倾向', 'Evidence_Type': '证据类型', 'Attribution_Level': '归因层次',
            'Temporal_Focus': '时间聚焦', 'Primary_Actor_Type': '主要行动者类型',
            'Geographic_Scope': '地理范围', 'Relationship_Model_Definition': '关系模式界定',
            'Discourse_Type': '话语类型',
            # 检验字段
            'Inspector_Is_CN_RU_Related': '人工检验：是否中俄相关',
            'Inspector_Boundary': '人工检验：边界划分',
            # VK特有
            'post_id': '帖子ID', 'comment_id': '评论ID', 'relevance': '相关性判断', 'channel_name': '频道名称'
        },
        'ru': {
            # 基础字段
            'Unit_ID': 'Идентификатор тематической единицы', 'Source': 'Источник данных', 
            'Post_ID': 'Идентификатор поста', 'Comment_ID': 'Идентификатор комментария',
            'Answer_ID': 'Идентификатор ответа', 'Unit_Text': 'Текст тематической единицы', 'Unit_Hash': 'Хеш текста',
            'speaker': 'Говорящий', 'Incident': 'Основное событие', 'expansion_logic': 'Логика разбивки',
            'processing_status': 'Статус обработки', 'comment_text': 'Текст комментария',
            'Post_Text': 'Текст поста', 'Comment_Text': 'Текст комментария', 'Question': 'Вопрос Знаху', 'Answer_Text': 'Текст ответа Знаху', 'Author': 'Автор',
            'Original_Post_ID': 'Оригинальный ID поста', 'Original_Comment_ID': 'Оригинальный ID комментария',
            # 分析维度
            'Valence': 'Эмоциональная окраска', 'Evidence_Type': 'Тип доказательств',
            'Attribution_Level': 'Уровень атрибуции', 'Temporal_Focus': 'Временной фокус',
            'Primary_Actor_Type': 'Тип основного актора', 'Geographic_Scope': 'Географический охват',
            'Relationship_Model_Definition': 'Определение модели отношений', 'Discourse_Type': 'Тип дискурса',
            # 检验字段
            'Inspector_Is_CN_RU_Related': 'Проверка экспертом: связано с КНР-РФ',
            'Inspector_Boundary': 'Проверка экспертом: границы',
            # VK特有
            'post_id': 'ID поста', 'comment_id': 'ID комментария', 'relevance': 'Оценка релевантности', 'channel_name': 'Название канала'
        }
    }
    
    @classmethod
    def _generate_frame_mappings(cls, lang):
        """动态生成框架相关映射"""
        frames = ['ProblemDefinition', 'ResponsibilityAttribution', 'MoralEvaluation', 
                 'SolutionRecommendation', 'ActionStatement', 'CausalExplanation']
        mappings = {}
        
        if lang == 'zh':
            frame_names = {'ProblemDefinition': '问题建构', 'ResponsibilityAttribution': '责任归因',
                          'MoralEvaluation': '道德评价', 'SolutionRecommendation': '解决方案',
                          'ActionStatement': '行动声明', 'CausalExplanation': '因果解释'}
            for frame in frames:
                cn_name = frame_names[frame]
                mappings[f'AI_Frame_{frame}_Present'] = f'AI识别：{cn_name}框架'
                mappings[f'Inspector_Frame_{frame}_Present'] = f'人工检验：{cn_name}框架'
        else:  # ru
            frame_names = {'ProblemDefinition': 'постановки проблемы', 'ResponsibilityAttribution': 'атрибуции ответственности',
                          'MoralEvaluation': 'моральной оценки', 'SolutionRecommendation': 'рекомендации решений',
                          'ActionStatement': 'заявления о действиях', 'CausalExplanation': 'причинного объяснения'}
            for frame in frames:
                ru_name = frame_names[frame]
                mappings[f'AI_Frame_{frame}_Present'] = f'ИИ распознал: фрейм {ru_name}'
                mappings[f'Inspector_Frame_{frame}_Present'] = f'Проверка экспертом: фрейм {ru_name}'
        
        return mappings
    
    @classmethod
    def _generate_dimension_mappings(cls, lang):
        """动态生成维度检验映射"""
        dims = ['Valence', 'Evidence_Type', 'Attribution_Level', 'Temporal_Focus',
               'Primary_Actor_Type', 'Geographic_Scope', 'Relationship_Model_Definition', 'Discourse_Type']
        mappings = {}
        
        prefix = '人工检验：' if lang == 'zh' else 'Проверка экспертом: правильность '
        suffix = '正确性' if lang == 'zh' else ''
        
        for dim in dims:
            base_name = cls._BASE_MAPPINGS[lang].get(dim, dim)
            mappings[f'Inspector_{dim}_Correct'] = f'{prefix}{base_name}{suffix}'
        
        return mappings
    
    @classmethod
    def get_mappings(cls, lang):
        """获取完整的映射字典"""
        mappings = cls._BASE_MAPPINGS[lang].copy()
        mappings.update(cls._generate_frame_mappings(lang))
        mappings.update(cls._generate_dimension_mappings(lang))
        return mappings
    
    # 兼容性属性
    @property
    def LABEL_MAPPINGS(self):
        return {'zh': self.get_mappings('zh'), 'ru': self.get_mappings('ru')}
    
    @staticmethod
    def decorate_headers(df: pd.DataFrame, lang: str) -> pd.DataFrame:
        """装饰列名为语言标签格式：本地语言名(英文名)"""
        mapping = BilingualSupport.get_mappings(lang)
        df_out = df.copy()
        new_columns = []
        for c in df.columns:
            if c in mapping:
                # 格式：本地语言名(英文名)
                new_columns.append(f"{mapping[c]}({c})")
            else:
                # 未映射的列名保持原样
                new_columns.append(c)
        df_out.columns = new_columns
        return df_out

# 保持向后兼容性的别名
_decorate_headers = BilingualSupport.decorate_headers
_decorate_headers_chinese = lambda df: BilingualSupport.decorate_headers(df, 'zh')
_decorate_headers_russian = lambda df: BilingualSupport.decorate_headers(df, 'ru')

def _clean_dataframe(df: pd.DataFrame, operation_name: str = "数据处理") -> pd.DataFrame:
    """通用的DataFrame清理函数：修复重复索引和列名"""
    # 重置索引
    df = df.reset_index(drop=True)
    
    # 检查并修复重复列名
    if df.columns.duplicated().any():
        UX.warn(f"{operation_name}发现重复列名: {df.columns[df.columns.duplicated()].tolist()}")
        df = df.loc[:, ~df.columns.duplicated()]
        UX.info(f"已去除重复列，当前列数: {len(df.columns)}")
    
    return df

def _is_frame_present(frame_value):
    """
    判断V2格式的框架内容是否真实存在。
    V2格式：[{"quote": "...", "reason": "...", "confidence": "..."}]
    """
    # 1. 处理Pandas的空值 (None, nan, etc.)
    if pd.isna(frame_value):
        return 0
    
    # 2. 如果是字符串，尝试解析
    if isinstance(frame_value, str):
        s_value = frame_value.strip()
        # 检查代表"空"或"无内容"的常见字符串
        if not s_value or s_value == '[]' or s_value == '[""]':
            return 0
        # 检查是否为明确的失败或不相关标记
        if 'API_FAILED' in s_value or 'NO_RELEVANT' in s_value:
            return 0
        try:
            frame_value = json.loads(s_value)
        except:
            return 0
    
    # 3. 如果是列表，检查是否有有效的对象
    if isinstance(frame_value, list):
        if not frame_value:  # 空列表
            return 0
        # 检查是否有包含有效quote的对象
        for item in frame_value:
            if isinstance(item, dict) and item.get('quote', '').strip():
                return 1
        return 0
    
    # 4. 其他情况认为无内容
    return 0

def _add_frame_boolean_columns(df, frames):
    """为DataFrame添加框架布尔值列（V2格式）"""
    for frame in frames:
        frame_col = f'Frame_{frame}'  # V2格式直接使用Frame_{frame}
        ai_col = f'AI_Frame_{frame}_Present'
        
        if frame_col in df.columns:
            df[ai_col] = df[frame_col].apply(_is_frame_present)
        else:
            df[ai_col] = 0
    
    return df

def _add_inspector_columns(df, frames, dims):
    """为DataFrame添加人工检验列"""
    # 添加框架检验列
    for frame in frames:
        inspector_col = f'Inspector_Frame_{frame}_Present'
        if inspector_col not in df.columns:
            df[inspector_col] = ''
    
    # 添加维度检验列
    for dim in dims:
        inspector_col = f'Inspector_{dim}_Correct'
        if inspector_col not in df.columns:
            df[inspector_col] = ''
    
    return df

def _organize_columns_order(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """
    根据数据源，使用预定义的顺序和范围来组织DataFrame的列。
    """
    # 理想的列顺序 master list
    base_order = [
        'Unit_ID', 'Unit_Hash', 'Source', 'Post_ID', 'Answer_ID', 
        'Post_Text', 'Unit_Text', 'speaker', 'Incident', 'expansion_logic',
        # 检验字段
        'Inspector_Boundary', 'Inspector_Is_CN_RU_Related'
    ]
    
    frames = ['ProblemDefinition', 'ResponsibilityAttribution', 'MoralEvaluation',
              'SolutionRecommendation', 'ActionStatement', 'CausalExplanation']
    dims = ['Valence', 'Evidence_Type', 'Attribution_Level', 'Temporal_Focus',
            'Primary_Actor_Type', 'Geographic_Scope', 'Relationship_Model_Definition',
            'Discourse_Type']

    for frame in frames:
        base_order.extend([f'Frame_{frame}', f'AI_Frame_{frame}_Present', f'Inspector_Frame_{frame}_Present'])
    
    for dim in dims:
        base_order.extend([dim, f'Inspector_{dim}_Correct'])

    # 定义每个数据源实际需要的列
    if source == 'vk':
        required_cols = {
            'Unit_ID', 'Unit_Hash', 'Source', 'Post_ID', 'Post_Text', 'Unit_Text', 
            'speaker', 'Incident', 
        }
    elif source == 'zhihu':
        required_cols = {
            'Unit_ID', 'Unit_Hash', 'Source', 'Answer_ID', 'Unit_Text', 
            'speaker', 'Incident', 'expansion_logic'
        }
    else: # 默认情况
        required_cols = set(df.columns)

    # 添加所有框架和维度的列到必需集合中
    for frame in frames:
        required_cols.update([f'Frame_{frame}', f'AI_Frame_{frame}_Present', f'Inspector_Frame_{frame}_Present'])
    for dim in dims:
        required_cols.update([dim, f'Inspector_{dim}_Correct'])
    
    # 1. 筛选出df中实际存在且必需的列
    existing_and_required = [col for col in df.columns if col in required_cols]
    
    # 2. 按照理想顺序对这些列进行排序
    final_ordered_cols = [col for col in base_order if col in existing_and_required]
    
    # 3. 添加任何不在理想顺序中但确实存在的必需列（以防万一）
    for col in existing_and_required:
        if col not in final_ordered_cols:
            final_ordered_cols.append(col)
            
    return df[final_ordered_cols]

def _save_bilingual(df: pd.DataFrame, zh_path: str, ru_path: str):
    """保存双语版本文件（优化策略：VK用俄语版，知乎用中文版）"""
    try:
        zh_dir = os.path.dirname(zh_path)
        ru_dir = os.path.dirname(ru_path)
        if zh_dir:
            os.makedirs(zh_dir, exist_ok=True)
        if ru_dir and ru_dir != zh_dir:
            os.makedirs(ru_dir, exist_ok=True)
    except Exception as e:
        UX.warn(f"创建输出目录失败: {e}")
    
    # 中文版：保持原样（适合知乎数据）
    try:
        df_zh = _decorate_headers(df, 'zh')
        df_zh.to_excel(zh_path, index=False)
    except Exception as e:
        UX.warn(f"中文版本导出失败: {e}")
    
    # 俄语版：保持原样（适合VK数据）
    try:
        df_ru = _decorate_headers(df, 'ru')
        df_ru.to_excel(ru_path, index=False)
    except Exception as e:
        UX.warn(f"俄语版本导出失败: {e}")

def _save_source_specific_bilingual(df: pd.DataFrame, output_path: str, file_prefix: str):
    """按信源分别保存双语版本（VK用俄语版，知乎用中文版）"""
    # 清理DataFrame
    df = _clean_dataframe(df, f"{file_prefix}保存")
    
    try:
        os.makedirs(output_path, exist_ok=True)
    except Exception as e:
        UX.warn(f"创建输出目录失败: {e}")
    
    # 按信源分组
    if 'Source' in df.columns:
        vk_data = df[df['Source'] == 'vk']
        zhihu_data = df[df['Source'] == '知乎']
        
        # VK数据：生成俄语版（内容本来就是俄语）
        if not vk_data.empty:
            ru_path = os.path.join(output_path, f'{file_prefix}_VK_俄语版.xlsx')
            try:
                df_ru = _organize_columns_order(vk_data, 'vk')
                df_ru = _decorate_headers(df_ru, 'ru')
                df_ru.to_excel(ru_path, index=False)
                UX.ok(f"VK俄语版已生成: {ru_path}")
            except Exception as e:
                UX.warn(f"VK俄语版导出失败: {e}")
        
        # 知乎数据：生成中文版（内容本来就是中文）
        if not zhihu_data.empty:
            zh_path = os.path.join(output_path, f'{file_prefix}_知乎_中文版.xlsx')
            try:
                df_zh = _organize_columns_order(zhihu_data, 'zhihu')
                df_zh = _decorate_headers(df_zh, 'zh')
                df_zh.to_excel(zh_path, index=False)
                UX.ok(f"知乎中文版已生成: {zh_path}")
            except Exception as e:
                UX.warn(f"知乎中文版导出失败: {e}")
        
    else:
        # 没有Source列，使用通用双语保存
        zh_path = os.path.join(output_path, f'{file_prefix}_中文版.xlsx')
        ru_path = os.path.join(output_path, f'{file_prefix}_俄语版.xlsx')
        _save_bilingual(df, zh_path, ru_path)

def generate_reliability_files_from_input(input_path, final_results_path, output_path):
    """直接从原始输入文件生成信度检验文件"""
    UX.info("从原始输入文件生成信度检验文件...")
    
    try:
        df_results = pd.read_excel(final_results_path)
        UX.info(f"最终结果数据库加载成功: {len(df_results)}条记录")
        UX.info(f"最终结果数据库列名: {list(df_results.columns)}")
    except Exception as e:
        UX.err(f"加载最终结果数据库失败: {e}")
        return
    
    # 获取输入文件
    files = [f for f in os.listdir(input_path) 
            if f.endswith('.xlsx') and not f.startswith('~$')]
    
    if not files:
        UX.warn("未找到输入文件")
        return
    
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
    
    # 构建父文本数据
    parent_data = []
    
    # 处理VK文件
    for filename in vk_files:
        input_file = os.path.join(input_path, filename)
        try:
            df = pd.read_excel(input_file)
            mapping = COLUMN_MAPPING['vk']
            
            for post_id, group in df.groupby(mapping['post_id'], dropna=False):
                if pd.isna(post_id):
                    continue
                    
                post_text = safe_str_convert(group[mapping['post_text']].iloc[0])
                # 获取channel_name，如果不存在则使用文件名
                channel_name = safe_str_convert(group[mapping.get('channel_name', 'channel_name')].iloc[0]) if mapping.get('channel_name', 'channel_name') in group.columns else filename.replace('.xlsx', '')
                
                # 为每个评论创建一条记录，包含post_text和comment_text
                for _, comment_row in group.iterrows():
                    comment_id = comment_row[mapping['comment_id']]
                    comment_text = comment_row[mapping['comment_text']]
                    
                    if pd.isna(comment_id) or pd.isna(comment_text):
                        continue
                    
                    parent_data.append({
                        'Post_ID': f"VK-{post_id}",
                        'Comment_ID': f"VK-{comment_id}",
                        'Original_Post_ID': str(post_id),
                        'Original_Comment_ID': str(comment_id),
                        'Source': 'vk',
                        'Post_Text': post_text,
                        'Comment_Text': safe_str_convert(comment_text),
                        'channel_name': channel_name
                    })
                
        except Exception as e:
            UX.warn(f"处理VK文件 {filename} 失败: {e}")
    
    # 处理知乎文件 - 基于结果数据构建父文本（包含议题单元）
    for filename in zhihu_files:
        input_file = os.path.join(input_path, filename)
        try:
            df = pd.read_excel(input_file)
            mapping = COLUMN_MAPPING['zhihu']
            
            # 从结果数据中获取知乎的议题单元
            zhihu_results = df_results[df_results['Source'] == '知乎']
            UX.info(f"知乎文件 {filename}: 结果数据中找到 {len(zhihu_results)} 条知乎议题单元")
            if not zhihu_results.empty:
                for _, unit_row in zhihu_results.iterrows():
                    unit_text = safe_str_convert(unit_row.get('Unit_Text', ''))
                    if not unit_text.strip():
                        continue
                    
                    # 从Unit_ID中提取原始answer_id
                    unit_id = str(unit_row.get('Unit_ID', ''))
                    if unit_id.startswith('ZH-') and '-' in unit_id:
                        original_answer_id = unit_id.split('-')[1]  # 提取answer_id部分
                        
                        # 从原始输入文件中找到对应的回答信息
                        original_row = df[df[mapping["id"]].astype(str) == original_answer_id]
                        if not original_row.empty:
                            original_row = original_row.iloc[0]
                            question = safe_str_convert(original_row[mapping["question"]])
                            # 智能获取作者名称
                            author_column = mapping.get("author", "回答用户名")
                            if author_column in original_row.index:
                                author_value = safe_str_convert(original_row[author_column])
                                author = author_value if author_value.strip() else '未知作者'
                            else:
                                author = '未知作者'
                            
                            parent_data.append({
                                'Answer_ID': f"ZH-{original_answer_id}",
                                'Original_ID': original_answer_id,
                                'Source': '知乎',
                                'Question': question,
                                'Unit_Text': unit_text,  # 使用议题单元文本而不是完整回答
                                'Author': author
                            })
            else:
                # 如果结果数据中没有知乎数据，回退到原始方式
                UX.warn(f"结果数据中没有知乎数据，跳过 {filename}")
                
        except Exception as e:
            UX.warn(f"处理知乎文件 {filename} 失败: {e}")
    
    if not parent_data:
        UX.warn("没有收集到父文本数据")
        return
    
    # 创建父文本DataFrame
    df_parent = pd.DataFrame(parent_data)
    UX.info(f"收集到 {len(df_parent)} 条父文本数据")
    
    # 标记使用情况 - 使用哈希值匹配
    UX.info("使用哈希值匹配父文本使用情况...")
    
    # 从结果数据中获取已使用的哈希值
    if 'Unit_Hash' not in df_results.columns:
        UX.warn(f"无法在结果数据中找到Unit_Hash列，可用列: {list(df_results.columns)}")
        df_parent['Was_Used'] = False
        used_count = 0
        UX.info(f"父文本使用情况: {used_count}/{len(df_parent)} 条被使用")
        return
    
    used_hashes = set(df_results['Unit_Hash'].dropna().astype(str).unique())
    UX.info(f"从结果数据中找到 {len(used_hashes)} 个已使用的哈希值")
    
    # 为父文本数据添加哈希值并匹配
    if 'Comment_Text' in df_parent.columns:
        # VK数据：基于Comment_Text生成哈希值
        df_parent['Text_Hash'] = df_parent['Comment_Text'].apply(
            lambda x: hashlib.sha256(normalize_text(safe_str_convert(x)).encode('utf-8')).hexdigest() 
            if pd.notna(x) and str(x).strip() else None
        )
        UX.info(f"VK数据使用评论文本哈希值匹配")
    elif 'Unit_Text' in df_parent.columns:
        # 知乎数据：基于Unit_Text生成哈希值
        df_parent['Text_Hash'] = df_parent['Unit_Text'].apply(
            lambda x: hashlib.sha256(normalize_text(safe_str_convert(x)).encode('utf-8')).hexdigest() 
            if pd.notna(x) and str(x).strip() else None
        )
        UX.info(f"知乎数据使用议题单元文本哈希值匹配")
    else:
        UX.warn("无法找到文本列用于哈希值匹配")
        df_parent['Was_Used'] = False
        used_count = 0
        UX.info(f"父文本使用情况: {used_count}/{len(df_parent)} 条被使用")
        return
    
    # 统一哈希值匹配
    df_parent['Was_Used'] = df_parent['Text_Hash'].astype(str).isin(used_hashes)
    
    used_count = df_parent['Was_Used'].sum()
    UX.info(f"父文本使用情况: {used_count}/{len(df_parent)} 条被使用")
    
    # 反向检验（召回率）- 只针对VK数据，知乎默认全相关
    negative_samples = []
    # 只处理VK数据
    if 'vk' in RELIABILITY_SAMPLING_CONFIG:
        cfg = RELIABILITY_SAMPLING_CONFIG['vk']
        unused = df_parent[(df_parent['Source'] == 'vk') & (df_parent['Was_Used'] == False)]
        UX.info(f"VK信源: 未使用样本 {len(unused)} 条，需要抽样 {cfg['recall']} 条")
        if len(unused) > 0:
            n = min(cfg['recall'], len(unused))
            if n > 0:
                sample = unused.sample(n=n, replace=False, random_state=2025)
                negative_samples.append(sample)
                UX.info(f"VK信源: 成功抽样 {len(sample)} 条反向检验样本")
            else:
                UX.warn(f"VK信源: 需要抽样数量为0")
        else:
            UX.warn(f"VK信源: 没有未使用的样本可供抽样")
    
    if negative_samples:
        df_neg = pd.concat(negative_samples, ignore_index=True)
        
        # 反向检验只保留VK相关字段（去掉Post_ID和Comment_ID）
        negative_essential_columns = ['Original_Post_ID', 'Original_Comment_ID', 'Source', 'Post_Text', 'Comment_Text', 'channel_name']
        
        # 过滤列（只保留存在的列）
        available_neg_columns = [col for col in negative_essential_columns if col in df_neg.columns]
        df_neg_clean = df_neg[available_neg_columns].copy()
        
        # 添加相关性检验列
        if 'Inspector_Is_CN_RU_Related' not in df_neg_clean.columns:
            df_neg_clean['Inspector_Is_CN_RU_Related'] = ''
        
        # 只生成VK俄语版（因为反向检验只涉及VK）
        try:
            os.makedirs(output_path, exist_ok=True)
            ru_path = os.path.join(output_path, '反向检验_召回率样本_VK_俄语版.xlsx')
            df_ru = _decorate_headers(df_neg_clean, 'ru')
            df_ru.to_excel(ru_path, index=False)
            UX.ok(f"VK反向检验样本已生成: {ru_path}")
        except Exception as e:
            UX.warn(f"VK反向检验样本导出失败: {e}")
    else:
        UX.warn("没有VK反向检验样本可供生成")
    
    # 框架维度检验（基于所有成功处理的结果）
    UX.info("生成框架维度检验文件...")
    
    # 获取所有成功处理的结果
    all_success_results = df_results[df_results['processing_status'] == ProcessingStatus.SUCCESS]
    
    if not all_success_results.empty:
        # 直接使用所有成功结果，让后续的列处理逻辑自动过滤
        df_combined = all_success_results.copy()
        
        # 简化：只在有VK数据时添加VK特有检验列
        if 'Source' in df_combined.columns and (df_combined['Source'] == 'vk').any():
            if 'Inspector_Boundary' not in df_combined.columns:
                df_combined['Inspector_Boundary'] = ''
            if 'Inspector_Is_CN_RU_Related' not in df_combined.columns:
                df_combined['Inspector_Is_CN_RU_Related'] = ''
            
            # 确保VK数据有Post_Text列（从原始输入文件获取）
            vk_mask = df_combined['Source'] == 'vk'
            if vk_mask.any():
                # 检查VK数据是否已经有有效的Post_Text列
                needs_post_text = True
                if 'Post_Text' in df_combined.columns:
                    # 检查Post_Text列是否有有效内容
                    vk_post_text_sample = df_combined[vk_mask]['Post_Text'].dropna()
                    if not vk_post_text_sample.empty and not vk_post_text_sample.iloc[0] in ['', '未找到原始帖子文本']:
                        needs_post_text = False
                        UX.info("VK数据已有有效的Post_Text列，跳过重新获取")
                
                if needs_post_text:
                    UX.info("开始为VK数据添加Post_Text列...")
                    
                    # 从原始输入文件获取Post_Text，使用哈希值匹配
                    comment_to_post_map = {}  # 评论哈希 -> 帖子文本
                    for filename in [f for f in os.listdir(input_path) if f.endswith('.xlsx') and not f.startswith('~$')]:
                        if 'vk' in filename.lower():
                            try:
                                input_file = os.path.join(input_path, filename)
                                df_input = pd.read_excel(input_file)
                                mapping = COLUMN_MAPPING['vk']
                                
                                for post_id, group in df_input.groupby(mapping['post_id'], dropna=False):
                                    if not pd.isna(post_id):
                                        post_text = safe_str_convert(group[mapping['post_text']].iloc[0])
                                        
                                        # 为每个评论计算哈希值并映射到帖子文本
                                        for _, row in group.iterrows():
                                            comment_text = safe_str_convert(row[mapping['comment_text']])
                                            if comment_text.strip():
                                                comment_norm = normalize_text(comment_text)
                                                comment_hash = hashlib.sha256(comment_norm.encode('utf-8')).hexdigest()
                                                comment_to_post_map[comment_hash] = post_text
                                
                                UX.info(f"从文件 {filename} 建立了 {len([k for k in comment_to_post_map.keys()])} 个评论到帖子的映射")
                            except Exception as e:
                                UX.warn(f"读取VK输入文件 {filename} 失败: {e}")
                
                    UX.info(f"总共建立了 {len(comment_to_post_map)} 个评论哈希到帖子文本的映射")
                    
                    # 为VK数据添加Post_Text列
                    if 'Post_Text' not in df_combined.columns:
                        df_combined['Post_Text'] = ''
                    
                    vk_data_count = vk_mask.sum()
                    UX.info(f"需要处理 {vk_data_count} 条VK数据")
                    
                    # 检查VK数据的列名
                    if vk_mask.any():
                        vk_sample = df_combined[vk_mask].iloc[0]
                        UX.info(f"VK数据列名: {list(vk_sample.index)}")
                    
                    success_count = 0
                    for idx, row in df_combined[vk_mask].iterrows():
                        # 使用Unit_Hash匹配
                        if 'Unit_Hash' in row and not pd.isna(row['Unit_Hash']):
                            unit_hash = row['Unit_Hash']
                            if unit_hash in comment_to_post_map:
                                df_combined.loc[idx, 'Post_Text'] = comment_to_post_map[unit_hash]
                                success_count += 1
                            else:
                                df_combined.loc[idx, 'Post_Text'] = f'未找到哈希匹配 (hash: {unit_hash[:8]}...)'
                        else:
                            df_combined.loc[idx, 'Post_Text'] = '未找到Unit_Hash列'
                    
                    UX.info(f"成功匹配 {success_count}/{vk_data_count} 条VK数据的Post_Text")
        
        # 定义框架和维度
        frames = ['ProblemDefinition', 'ResponsibilityAttribution', 'MoralEvaluation',
                 'SolutionRecommendation', 'ActionStatement', 'CausalExplanation']
        dims = ['Valence', 'Evidence_Type', 'Attribution_Level', 'Temporal_Focus',
               'Primary_Actor_Type', 'Geographic_Scope', 'Relationship_Model_Definition',
               'Discourse_Type']
        
        # 使用新的公共函数处理
        df_combined = _add_frame_boolean_columns(df_combined, frames)
        df_combined = _add_inspector_columns(df_combined, frames, dims)
        # 注意：这里不调用 _organize_columns_order，因为数据是混合的
        # 列的组织将在 _save_source_specific_bilingual 中按信源分别处理
        
        # 清理并保存
        df_combined = _clean_dataframe(df_combined, "框架维度检验样本")
        _save_source_specific_bilingual(df_combined, output_path, '框架维度检验_单检验员')
        UX.ok("框架维度检验文件已生成(按信源分语言)")
    else:
        UX.warn("没有成功处理的结果可供生成框架维度检验文件")
    
    UX.ok("信度检验文件生成完成")

# ==============================================================================
# === 📊 主函数
# ==============================================================================

async def main():
    """主函数"""
    UX.start_run()
    UX.phase("社交媒体议题单元分析器启动")
    UX.info(f"信度检验模式: {'开启' if RELIABILITY_TEST_MODE else '关闭'}")
    
    # 检查配置
    required_models = ["VK_BATCH", "ZHIHU_CHUNKING", "ZHIHU_ANALYSIS"]
    missing = [m for m in required_models if m not in API_CONFIG.get("STAGE_MODELS", {})]
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
    api_service_sync = APIService()
    
    # 处理VK文件
    if vk_files:
        UX.phase("处理VK文件")
        
        for filename in vk_files:
            input_file = os.path.join(INPUT_PATH, filename)
            output_file = os.path.join(OUTPUT_PATH, f"(不能删)analyzed_{filename}")
            
            try:
                df = pd.read_excel(input_file)
                processor = VKProcessor(api_service_sync)
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
            api_service_async = APIService(session)
            
            for filename in zhihu_files:
                input_file = os.path.join(INPUT_PATH, filename)
                output_file = os.path.join(OUTPUT_PATH, f"(不能删)analyzed_{filename}")
                
                try:
                    df = pd.read_excel(input_file)
                    processor = ZhihuProcessor(api_service_async)
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
        
        # 生成信度检验文件（直接从原始输入文件）
        if final_path:
            try:
                UX.info("开始生成信度检验文件...")
                generate_reliability_files_from_input(INPUT_PATH, final_path, OUTPUT_PATH)
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