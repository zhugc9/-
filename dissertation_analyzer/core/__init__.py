# -*- coding: utf-8 -*-
"""
Core module for media and social media text analysis system.
"""

__version__ = "1.0.0"
__author__ = "Media Social Media Text Analyzer"

# Core components
from .utils import (
    UX, load_config, get_tokenizer, count_tokens, safe_str_convert, normalize_text,
    detect_language_and_get_config, identify_source,
    get_processing_state, create_unified_record,
    ProcessingStatus, ANALYSIS_COLUMNS
)
from .api_service import APIService
from .processors import BaseProcessor, MediaTextProcessor, VKProcessor, ZhihuProcessor
from .reliability import ReliabilityTestModule
from .auto_retry import AutoRetryManager, create_auto_retry_manager

__all__ = [
    'UX',
    'load_config', 
    'get_tokenizer',
    'count_tokens',
    'safe_str_convert',
    'normalize_text',
    'detect_language_and_get_config',
    'identify_source',
    'get_processing_state',
    'create_unified_record',
    'ProcessingStatus',
    'ANALYSIS_COLUMNS',
    'APIService',
    'BaseProcessor',
    'MediaTextProcessor', 
    'VKProcessor',
    'ZhihuProcessor',
    'ReliabilityTestModule',
    'AutoRetryManager',
    'create_auto_retry_manager'
]
