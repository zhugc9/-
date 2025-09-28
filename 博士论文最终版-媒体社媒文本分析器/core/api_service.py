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

    def _select_model(self, stage_key: str, explicit_model: str = None) -> str:
        if explicit_model:
            return explicit_model
        
        # 获取主模型和备用模型
        model_pools = self.config.get('model_pools', {})
        if model_pools and 'primary_models' in model_pools:
            primary = model_pools['primary_models'].get(stage_key)
            fallback_list = model_pools.get('fallback_models', {}).get(stage_key, [])
            fallback = fallback_list[0] if fallback_list else primary  # 没有备用就用主模型
        else:
            # 向后兼容：使用旧的配置
            api_config = self.config.get('api_config', {})
            primary = api_config.get('STAGE_MODELS', {}).get(stage_key)
            fallback_list = api_config.get('FALLBACK', {}).get('STAGE_CANDIDATES', {}).get(stage_key, [])
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
        max_model_switches = self.config.get('api_retry_config', {}).get('max_model_switches', 10)
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
        
        # 从配置读取语言配置和API配置
        language_configs = self.config.get('LANGUAGE_CONFIGS', {})
        config = language_configs.get(language, language_configs.get('zh', {}))
        prompt_tokens = count_tokens(prompt)
        
        # 从统一超时配置读取token阈值和超时时间
        timeout_config = self.config.get('TIMEOUT_CONFIG', {})
        threshold_short = timeout_config.get('TOKEN_THRESHOLD_SHORT', 2000)
        threshold_medium = timeout_config.get('TOKEN_THRESHOLD_MEDIUM', 4000)
        
        if prompt_tokens < threshold_short:
            timeout = timeout_config.get('TIMEOUT_SHORT', 400)  # 短文本
        elif prompt_tokens <= threshold_medium:
            timeout = timeout_config.get('TIMEOUT_MEDIUM', 500)  # 中等文本
        else:
            timeout = timeout_config.get('TIMEOUT_LONG', 500)  # 长文本
        
        # 从配置读取重试设置
        api_retry_config = self.config.get('api_retry_config', {})
        api_retry_attempts = api_retry_config.get('attempts_per_model', 3)
        retry_delays = api_retry_config.get('retry_delays', [2, 5, 10])
        
        for attempt in range(api_retry_attempts):
            try:
                api_config = self.config.get('api_config', {})
                api_key = api_config.get("API_KEYS", [""])[0]
                url = f"{api_config.get('BASE_URL', '')}/chat/completions"
                headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
                chosen_model = self._select_model(stage_key or 'INTEGRATED_ANALYSIS', model_name)
                
                # 从配置读取API请求参数
                api_request_params = self.config.get('api_request_params', {})
                payload = {
                    "model": chosen_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": api_request_params.get('temperature', 0),
                    "response_format": api_request_params.get('response_format', {"type": "json_object"})
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


# 同步版本的APIService，用于兼容社交媒体项目
class SyncAPIService:
    """同步版本的API服务，用于兼容需要同步调用的场景"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.call_count = 0
        self.success_count = 0
        self.failure_count = 0
    
    def _create_payload(self, prompt, stage_key):
        """创建API请求负载"""
        api_config = self.config.get('api_config', {})
        stage_models = api_config.get('STAGE_MODELS', {})
        
        return {
            "model": stage_models.get(stage_key, "[官自-0.7]gemini-2-5-flash"),
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
    
    def call_api_sync(self, prompt, language='zh', stage_key=None):
        """同步API调用"""
        import requests
        self.call_count += 1
        
        language_configs = self.config.get('LANGUAGE_CONFIGS', {})
        config = language_configs.get(language, language_configs.get('zh', {}))
        
        api_retry_attempts = self.config.get('API_RETRY_ATTEMPTS', 3)
        rate_limit_base_delay = self.config.get('RATE_LIMIT_BASE_DELAY', 2)
        
        for attempt in range(api_retry_attempts):
            try:
                api_config = self.config.get('API_CONFIG', {})
                url = f"{api_config['BASE_URL']}/chat/completions"
                headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_config['API_KEYS'][0]}"}
                payload = self._create_payload(prompt, stage_key)
                
                response = requests.post(url, headers=headers, json=payload, timeout=config.get('TIMEOUT', 180))
                response.raise_for_status()
                content = response.json()['choices'][0]['message']['content']
                
                result = self._extract_json_response(content)
                self.success_count += 1
                return result
                
            except Exception as e:
                if attempt < api_retry_attempts - 1:
                    import time
                    time.sleep(rate_limit_base_delay * (2 ** attempt))
                else:
                    self.failure_count += 1
                    skip_failed_texts = self.config.get('SKIP_FAILED_TEXTS', True)
                    if skip_failed_texts:
                        UX.warn(f"API失败: {str(e)[:100]}")
                        return None
                    raise
        return None
