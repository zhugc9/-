# -*- coding: utf-8 -*-
"""
API service module for handling all external API calls.
"""

import json
import asyncio
import aiohttp
from .utils import UX, count_tokens

class APIService:
    def __init__(self, session, config=None):
        self.session = session
        self.config = config or {}
        self.fail_counts = {}
        self.model_index = {}
        self.total_switches = {}
        # API统计计数器
        self.call_count = 0
        self.success_count = 0
        self.failure_count = 0

    def _select_model(self, stage_key: str, explicit_model: str = None) -> str:
        if explicit_model:
            return explicit_model
        
        # 优先尝试从专用配置获取模型
        primary = None
        
        # 媒体文本模型：从media_text.model_pools获取
        if stage_key in ['UNIT_EXTRACTION', 'UNIT_ANALYSIS']:
            media_config = self.config.get('media_text', {})
            model_pools = media_config.get('model_pools', {})
            if model_pools and 'primary_models' in model_pools:
                primary = model_pools['primary_models'].get(stage_key)
        
        # 社交媒体模型：从social_media.model_pools.primary_models获取
        elif stage_key in ['VK_BATCH', 'VK_BATCH_LONG', 'ZHIHU_CHUNKING', 'ZHIHU_ANALYSIS_SHORT', 'ZHIHU_ANALYSIS', 'ZHIHU_ANALYSIS_LONG']:
            social_config = self.config.get('social_media', {})
            model_pools = social_config.get('model_pools', {})
            if model_pools and 'primary_models' in model_pools:
                primary = model_pools['primary_models'].get(stage_key)
        
        # 获取备用模型
        fallback = primary
        if stage_key in ['UNIT_EXTRACTION', 'UNIT_ANALYSIS']:
            # 媒体文本模型：从media_text.model_pools.fallback_models获取备用模型
            media_config = self.config.get('media_text', {})
            model_pools = media_config.get('model_pools', {})
            if model_pools:
                fallback_list = model_pools.get('fallback_models', {}).get(stage_key, [])
                # 如果备用模型为空字符串，则不使用备用模型，继续使用主模型
                fallback = fallback_list[0] if fallback_list and fallback_list[0].strip() else primary
        elif stage_key in ['VK_BATCH', 'VK_BATCH_LONG', 'ZHIHU_CHUNKING', 'ZHIHU_ANALYSIS_SHORT', 'ZHIHU_ANALYSIS', 'ZHIHU_ANALYSIS_LONG']:
            # 社交媒体模型：从social_media.model_pools.fallback_models获取备用模型
            social_config = self.config.get('social_media', {})
            model_pools = social_config.get('model_pools', {})
            if model_pools:
                fallback_list = model_pools.get('fallback_models', {}).get(stage_key, [])
                # 如果备用模型为空字符串，则不使用备用模型，继续使用主模型
                fallback = fallback_list[0] if fallback_list and fallback_list[0].strip() else primary
        
        if not primary:
            # 向后兼容：尝试旧的配置结构
            model_pools = self.config.get('model_pools', {})
            if model_pools and 'primary_models' in model_pools:
                primary = model_pools['primary_models'].get(stage_key)
                fallback_list = model_pools.get('fallback_models', {}).get(stage_key, [])
                # 如果备用模型为空字符串，则不使用备用模型，继续使用主模型
                fallback = fallback_list[0] if fallback_list and fallback_list[0].strip() else primary
            else:
                # 向后兼容：从社交媒体配置中获取模型
                social_config = self.config.get('social_media', {})
                social_models = social_config.get('model_pools', {}).get('primary_models', {})
                primary = social_models.get(stage_key)
                fallback_list = social_config.get('model_pools', {}).get('fallback_models', {}).get(stage_key, [])
                # 如果备用模型为空字符串，则不使用备用模型，继续使用主模型
                fallback = fallback_list[0] if fallback_list and fallback_list[0].strip() else primary
        
        # 如果主备相同，就不切换
        if primary == fallback:
            return primary
        
        models = [primary, fallback]
        idx = self.model_index.get(stage_key, 0)
        return models[idx]

    def _handle_failure(self, stage_key: str) -> bool:
        """3次失败后切换模型"""
        # 切换计数
        api_strategy = self.config.get('api', {}).get('strategy', {})
        max_model_switches = api_strategy.get('max_model_switches', 10)
        self.total_switches[stage_key] = self.total_switches.get(stage_key, 0) + 1
        
        # 检查是否达到上限
        if self.total_switches[stage_key] >= max_model_switches:
            return False
        
        # 切换模型：主模型 ←→ 备用模型
        self.model_index[stage_key] = 1 - self.model_index.get(stage_key, 0)
        UX.info(f"[{stage_key}] 切换模型 (第{self.total_switches[stage_key]}次)")
        return True

    async def _call_api(self, prompt, language='zh', model_name=None, stage_key=None, context_label=None):
        # 初始化变量，避免在异常处理中引用未定义的变量
        response = None
        url = ""
        
        # 语言参数保留（用于模型选择等），但不再用于超时和并发配置
        prompt_tokens = count_tokens(prompt)
        
        # 使用统一的超时配置
        api_config = self.config.get('api', {})
        api_strategy = api_config.get('strategy', {})
        timeout = api_strategy.get('timeout_sec', 180)
        
        # 从新配置结构读取重试设置
        api_strategy = api_config.get('strategy', {})
        api_retry_attempts = api_strategy.get('attempts_per_model', 3)
        retry_delays = api_strategy.get('retry_delays_sec', [2, 5, 10])
        
        for attempt in range(api_retry_attempts):
            try:
                # 从新配置结构获取API凭据
                credentials = api_config.get('credentials', {})
                api_key = credentials.get('keys', [''])[0]
                url = f"{credentials.get('base_url', '')}/chat/completions"
                headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
                chosen_model = self._select_model(stage_key or 'INTEGRATED_ANALYSIS', model_name)
                
                # 从新配置结构读取API请求参数
                request_params = api_config.get('request_params', {})
                payload = {
                    "model": chosen_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": request_params.get('temperature', 0),
                    "response_format": request_params.get('response_format', {"type": "json_object"})
                }
                
                async with self.session.post(url, headers=headers, json=payload,
                                           timeout=aiohttp.ClientTimeout(total=timeout)) as response:
                    response_text = await response.text()
                    response.raise_for_status()
                    response_json = json.loads(response_text)
                    content = response_json.get('choices', [{}])[0].get('message', {}).get('content')
                    
                    if not content:
                        raise ValueError("响应缺少content")
                    
                    # 提取JSON（支持对象和数组）
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
                                result = json.loads(content[first_bracket:last_bracket+1])
                                # 成功重置失败计数
                                if stage_key:
                                    self.fail_counts[stage_key] = 0
                                return result
                            except json.JSONDecodeError:
                                pass
                    
                    if first_brace >= 0 and last_brace > first_brace:
                        # 是对象
                        try:
                            result = json.loads(content[first_brace:last_brace+1])
                            # 成功重置失败计数
                            if stage_key:
                                self.fail_counts[stage_key] = 0
                            return result
                        except json.JSONDecodeError:
                            pass
                    
                    # 最后尝试直接解析整个内容
                    try:
                        result = json.loads(content)
                        # 成功重置失败计数
                        if stage_key:
                            self.fail_counts[stage_key] = 0
                        return result
                    except json.JSONDecodeError:
                        pass
                    
                    raise ValueError("未找到有效JSON结构")
            
            except Exception as e:
                # 简要API失败日志
                status_code = response.status if response else 'N/A'
                error_brief = str(e)[:50] + "..." if len(str(e)) > 50 else str(e)
                UX.api_failed(stage_key or 'API', f"{status_code} - {error_brief}")
                
                # 重试延迟（除了最后一次）
                if attempt < api_retry_attempts - 1:
                    delay = retry_delays[attempt] if attempt < len(retry_delays) else 2
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
        
        # 如果expected_key为None，直接返回整个结果（用于第二阶段分析）
        if expected_key is None:
            return result
        
        if not isinstance(result, dict) or expected_key not in result:
            skip_failed_texts = self.config.get('data_processing', {}).get('skip_failed_texts', True)
            if skip_failed_texts:
                return None
            raise ValueError(f"返回JSON缺少键: '{expected_key}'")
        
        return result[expected_key]

    async def call_api_async(self, prompt, language='zh', stage_key=None):
        """兼容社交媒体项目的异步API调用方法"""
        # 统计计数
        self.call_count += 1
        
        # 直接调用内部_call_api方法
        result = await self._call_api(prompt, language, None, stage_key, None)
        
        if result is not None:
            self.success_count += 1
            return result
        else:
            self.failure_count += 1
            return None

    # call_api_sync方法已删除 - 现在直接使用SyncAPIService，避免冗余包装

    def print_statistics(self):
        """打印API调用统计"""
        if self.call_count > 0:
            success_rate = (self.success_count / self.call_count) * 100
            UX.info(f"API统计 - 总调用: {self.call_count}, 成功: {self.success_count}, "
                    f"失败: {self.failure_count}, 成功率: {success_rate:.1f}%")
        else:
            UX.info("API统计 - 无调用记录")

# 同步版本的APIService，用于兼容社交媒体项目
class SyncAPIService:
    """同步版本的API服务，支持模型轮换功能"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.call_count = 0
        self.success_count = 0
        self.failure_count = 0
        # 模型切换相关状态
        self.model_index = {}  # 跟踪每个stage_key的当前模型索引
        self.total_switches = {}  # 跟踪每个stage_key的总切换次数
    
    def _select_model(self, stage_key: str) -> str:
        """选择模型（支持轮换）"""
        # 优先尝试从社交媒体专用模型池获取
        social_config = self.config.get('social_media', {})
        model_pools = social_config.get('model_pools', {})
        
        if model_pools and 'primary_models' in model_pools:
            primary = model_pools['primary_models'].get(stage_key)
            if primary:
                # 获取备用模型
                fallback_list = model_pools.get('fallback_models', {}).get(stage_key, [])
                fallback = fallback_list[0] if fallback_list and fallback_list[0] else primary
                
                # 如果主备相同，直接返回
                if primary == fallback:
                    return primary
                
                # 根据当前索引选择模型
                models = [primary, fallback]
                idx = self.model_index.get(stage_key, 0)
                return models[idx]
        
        # 向后兼容：尝试社交媒体配置结构
        social_config = self.config.get('social_media', {})
        stage_models = social_config.get('model_pools', {}).get('primary_models', {})
        return stage_models.get(stage_key, "[官自-0.7]gemini-2-5-flash")
    
    def _handle_failure(self, stage_key: str) -> bool:
        """处理失败并切换模型"""
        api_strategy = self.config.get('api', {}).get('strategy', {})
        max_model_switches = api_strategy.get('max_model_switches', 2)
        
        self.total_switches[stage_key] = self.total_switches.get(stage_key, 0) + 1
        
        if self.total_switches[stage_key] >= max_model_switches:
            return False
        
        # 检查是否有备用模型可以切换
        social_config = self.config.get('social_media', {})
        model_pools = social_config.get('model_pools', {})
        
        if model_pools and 'primary_models' in model_pools:
            primary = model_pools['primary_models'].get(stage_key)
            fallback_list = model_pools.get('fallback_models', {}).get(stage_key, [])
            # 如果备用模型为空字符串，则不使用备用模型
            fallback = fallback_list[0] if fallback_list and fallback_list[0].strip() else None
            
            if primary and fallback and primary != fallback:
                # 切换模型：主模型 ←→ 备用模型
                self.model_index[stage_key] = 1 - self.model_index.get(stage_key, 0)
                UX.info(f"[{stage_key}] 切换模型 (第{self.total_switches[stage_key]}次)")
                return True
        
        UX.info(f"[{stage_key}] 记录失败，无备用模型 (第{self.total_switches[stage_key]}次)")
        return True
    
    def _create_payload(self, prompt, stage_key):
        """创建API请求负载"""
        chosen_model = self._select_model(stage_key)
        
        # 从新配置结构获取请求参数
        api_config = self.config.get('api', {})
        request_params = api_config.get('request_params', {})
        
        return {
            "model": chosen_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": request_params.get('temperature', 0),
            "response_format": request_params.get('response_format', {"type": "json_object"})
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
    
    def call_api_sync(self, prompt, language='zh', stage_key=None):
        """同步API调用，支持重试和模型轮换框架"""
        import requests
        import time
        self.call_count += 1
        
        # 获取配置
        api_config = self.config.get('api', {})
        api_strategy = api_config.get('strategy', {})
        attempts_per_model = api_strategy.get('attempts_per_model', 3)
        retry_delays = api_strategy.get('retry_delays_sec', [2, 5, 10])
        
        # 获取API凭据（支持新旧配置）
        credentials = api_config.get('credentials', {})
        api_key = credentials.get('keys', [''])[0] if credentials.get('keys') else None
        base_url = credentials.get('base_url', '')
        
        # 如果新配置不存在，则报错（不再向后兼容）
        if not api_key or not base_url:
            raise ValueError("API密钥和基础URL必须在config.yaml的api.credentials中配置")
        
        # 使用统一超时配置
        api_strategy = api_config.get('strategy', {})
        timeout = api_strategy.get('timeout_sec', 180)
        
        headers = {
            "Content-Type": "application/json", 
            "Authorization": f"Bearer {api_key}"
        }
        
        # 重试循环
        for attempt in range(attempts_per_model):
            try:
                payload = self._create_payload(prompt, stage_key)
                
                response = requests.post(
                    f"{base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=timeout
                )
                
                response.raise_for_status()
                content = response.json()['choices'][0]['message']['content']
                
                result = self._extract_json_response(content)
                if result:
                    self.success_count += 1
                    return result
                else:
                    UX.warn(f"[{stage_key}] 第{attempt+1}次尝试：JSON解析失败")
                
            except Exception as e:
                UX.warn(f"[{stage_key}] 第{attempt+1}次尝试异常: {str(e)[:100]}")
                
                # 如果不是最后一次尝试，等待后重试
                if attempt < attempts_per_model - 1:
                    delay = retry_delays[min(attempt, len(retry_delays)-1)]
                    time.sleep(delay)
        
        # 所有重试都失败了
        self.failure_count += 1
        self._handle_failure(stage_key)  # 记录失败
        
        # 检查是否跳过失败
        skip_failed = self.config.get('processing', {}).get('general', {}).get('skip_on_api_failure', True)
        if not skip_failed:
            skip_failed = self.config.get('SKIP_FAILED_TEXTS', True)  # 向后兼容
            
        if skip_failed:
            UX.warn(f"[{stage_key}] API调用失败，已尝试{attempts_per_model}次")
            return None
        else:
            raise Exception(f"API调用失败: {stage_key}")
        
        return None