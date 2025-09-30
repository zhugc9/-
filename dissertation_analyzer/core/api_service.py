# -*- coding: utf-8 -*-
"""
API service module for handling all external API calls.
"""

import json
import asyncio
import aiohttp
from .utils import UX

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

    def _get_model_candidates(self, stage_key: str):
        pools = self.config.get('model_pools', {})
        stage_config = pools.get(stage_key)
        if not stage_config:
            raise ValueError(f"未配置模型: {stage_key}")

        candidates = []
        primary = stage_config.get('primary')
        if primary and primary.strip():
            candidates.append(primary.strip())

        for fallback in stage_config.get('fallback', []) or []:
            if fallback and fallback.strip():
                candidates.append(fallback.strip())

        if not candidates:
            raise ValueError(f"模型{stage_key}未提供可用候选项")

        return candidates

    def _select_model(self, stage_key: str, explicit_model: str = None) -> str:
        if explicit_model:
            return explicit_model

        candidates = self._get_model_candidates(stage_key)
        idx = self.model_index.get(stage_key, 0)
        if idx >= len(candidates):
            idx = len(candidates) - 1
        return candidates[idx]

    def _handle_failure(self, stage_key: str) -> bool:
        """3次失败后切换模型"""
        api_strategy = self.config.get('api', {}).get('strategy', {})
        max_model_switches = api_strategy.get('max_model_switches', 10)
        self.total_switches[stage_key] = self.total_switches.get(stage_key, 0) + 1

        if self.total_switches[stage_key] >= max_model_switches:
            return False

        candidates = self._get_model_candidates(stage_key)
        current_idx = self.model_index.get(stage_key, 0)

        if current_idx < len(candidates) - 1:
            self.model_index[stage_key] = current_idx + 1
            UX.info(f"[{stage_key}] 切换模型 (第{self.total_switches[stage_key]}次)")
            return True

        return False

    async def _call_api(self, prompt, language='zh', model_name=None, stage_key=None, context_label=None):
        # 初始化变量，避免在异常处理中引用未定义的变量
        response = None
        url = ""
        
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
                                    self.model_index[stage_key] = 0
                                    self.total_switches[stage_key] = 0
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
                                self.model_index[stage_key] = 0
                                self.total_switches[stage_key] = 0
                            return result
                        except json.JSONDecodeError:
                            pass
                    
                    raise ValueError("未找到有效JSON结构")
            
            except Exception as e:
                status_code = response.status if response else 'N/A'
                error_brief = str(e)[:50] + "..." if len(str(e)) > 50 else str(e)
                UX.api_failed(stage_key or 'API', f"{status_code} - {error_brief}")

                if attempt < api_retry_attempts - 1:
                    delay = retry_delays[attempt] if attempt < len(retry_delays) else 2
                    await asyncio.sleep(delay)
        
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