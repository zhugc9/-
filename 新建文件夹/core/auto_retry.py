# -*- coding: utf-8 -*-
"""
自动重试模块 - 通用自动重试功能
支持媒体文本和社交媒体分析的统一重试机制
"""

import os
import glob
import asyncio
import pandas as pd
from .utils import UX, ProcessingStatus


class AutoRetryManager:
    """自动重试管理器"""
    
    def __init__(self, config, output_path):
        """
        初始化自动重试管理器
        
        Args:
            config: 配置字典，包含processing.auto_retry配置
            output_path: 输出文件路径，用于检查失败记录
        """
        self.config = config
        self.output_path = output_path
        
        # 从配置中获取重试参数
        retry_config = config.get('processing', {}).get('auto_retry', {})
        self.enabled = retry_config.get('enabled', False)
        self.max_rounds = retry_config.get('max_rounds', 5)
        self.delay_minutes = retry_config.get('delay_minutes', 2)
        self.min_failed_threshold = retry_config.get('min_failed_threshold', 1)
    
    def check_failed_records(self) -> tuple:
        """
        检查输出目录中是否存在API失败记录
        
        Returns:
            tuple: (总失败数量, 失败文件列表)
        """
        failed_files = []
        total_failed_count = 0
        
        try:
            # 查找所有analyzed结果文件
            analyzed_files = glob.glob(os.path.join(self.output_path, "*analyzed_*.xlsx"))
            
            for file_path in analyzed_files:
                try:
                    df = pd.read_excel(file_path)
                    if not df.empty and 'processing_status' in df.columns:
                        failed_mask = df['processing_status'].isin({
                            ProcessingStatus.STAGE_1_FAILED,
                            ProcessingStatus.STAGE_2_FAILED
                        })
                        failed_count = failed_mask.sum()
                        if failed_count > 0:
                            failed_files.append({
                                'file': os.path.basename(file_path),
                                'failed_count': failed_count
                            })
                            total_failed_count += failed_count
                except Exception as e:
                    UX.warn(f"检查文件失败记录时出错 {file_path}: {e}")
            
            return total_failed_count, failed_files
        
        except Exception as e:
            UX.err(f"检查失败记录时发生错误: {e}")
            return 0, []
    
    async def run_with_auto_retry(self, processing_func, *args, **kwargs):
        """
        执行带自动重试的处理逻辑
        
        Args:
            processing_func: 主处理函数（异步）
            *args, **kwargs: 传递给处理函数的参数
        """
        if not self.enabled:
            UX.info("自动重试模式: 关闭")
            return await processing_func(*args, **kwargs)
        
        UX.info("自动重试模式: 开启")
        retry_round = 0
        
        while retry_round < self.max_rounds:
            retry_round += 1
            
            # 标记轮次开始
            UX.phase(f"━━━ 第 {retry_round} 轮处理开始 ━━━")
            
            # 执行主处理逻辑
            result = await processing_func(*args, **kwargs)
            
            # 轮次结束标记
            UX.phase(f"━━━ 第 {retry_round} 轮处理完成 ━━━")
            
            # 检查是否还有失败记录
            total_failed, failed_files = self.check_failed_records()
            
            if total_failed >= self.min_failed_threshold:
                UX.warn(f"检测到 {total_failed} 条API失败记录，准备第 {retry_round + 1} 轮重试")
                for file_info in failed_files:
                    UX.info(f"   {file_info['file']}: {file_info['failed_count']} 条失败或待重试记录")
                
                if retry_round < self.max_rounds:
                    UX.info(f"等待 {self.delay_minutes} 分钟后开始重试...")
                    try:
                        await asyncio.sleep(self.delay_minutes * 60)  # 转换为秒
                    except asyncio.CancelledError:
                        UX.warn("重试等待被中断，停止自动重试")
                        raise  # 重新抛出，让上层处理
                else:
                    UX.warn(f"已达到最大重试轮数 ({self.max_rounds})，停止重试")
                    break
            else:
                if total_failed > 0:
                    UX.info(f"剩余 {total_failed} 条失败记录（小于阈值 {self.min_failed_threshold}），处理完成")
                else:
                    UX.ok("所有记录处理成功，无需重试")
                break
        
        UX.phase("自动重试流程完成")
        return result
    
    def run_with_auto_retry_sync(self, processing_func, *args, **kwargs):
        """
        执行带自动重试的处理逻辑（同步版本）
        
        Args:
            processing_func: 主处理函数（同步）
            *args, **kwargs: 传递给处理函数的参数
        """
        if not self.enabled:
            UX.info("自动重试模式: 关闭")
            return processing_func(*args, **kwargs)
        
        UX.info("自动重试模式: 开启")
        retry_round = 0
        
        while retry_round < self.max_rounds:
            retry_round += 1
            
            # 标记轮次开始
            UX.phase(f"━━━ 第 {retry_round} 轮处理开始 ━━━")
            
            # 执行主处理逻辑
            result = processing_func(*args, **kwargs)
            
            # 轮次结束标记
            UX.phase(f"━━━ 第 {retry_round} 轮处理完成 ━━━")
            
            # 检查是否还有失败记录
            total_failed, failed_files = self.check_failed_records()
            
            if total_failed >= self.min_failed_threshold:
                UX.warn(f"检测到 {total_failed} 条API失败记录，准备第 {retry_round + 1} 轮重试")
                for file_info in failed_files:
                    UX.info(f"   {file_info['file']}: {file_info['failed_count']} 条失败")
                
                if retry_round < self.max_rounds:
                    UX.info(f"等待 {self.delay_minutes} 分钟后开始重试...")
                    import time
                    time.sleep(self.delay_minutes * 60)  # 转换为秒
                else:
                    UX.warn(f"已达到最大重试轮数 ({self.max_rounds})，停止重试")
                    break
            else:
                if total_failed > 0:
                    UX.info(f"剩余 {total_failed} 条失败记录（小于阈值 {self.min_failed_threshold}），处理完成")
                else:
                    UX.ok("所有记录处理成功，无需重试")
                break
        
        UX.phase("自动重试流程完成")
        return result

def create_auto_retry_manager(config, output_path):

    return AutoRetryManager(config, output_path)