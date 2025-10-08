"""模块化的信度检验生成器"""

from __future__ import annotations

import os
import json
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import pandas as pd

try:
    from rapidfuzz import fuzz, utils
except ImportError:  # pragma: no cover
    from thefuzz import fuzz, utils  # type: ignore

from . import UX
from .utils import ProcessingStatus

# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------

_UNIT_TEXT_CANDIDATES = [
    "Unit_Text",
    "unit_text",
    "UnitText",
    "comment_text",
    "评论内容",
    "回答内容",
    "Answer_Text",
    "text",
    "正文",
]

_ID_CANDIDATES = [
    "序号",       # 媒体文本和知乎的文章/回答ID
    "Unit_ID",    # 单元唯一标识
    "comment_id", # VK评论ID
]


def _analysis_columns() -> List[str]:
    return [
        "Incident",
        "Frame_SolutionRecommendation",
        "Frame_ResponsibilityAttribution",
        "Frame_CausalExplanation",
        "Frame_MoralEvaluation",
        "Frame_ProblemDefinition",
        "Frame_ActionStatement",
        "Valence",
        "Evidence_Type",
        "Attribution_Level",
        "Temporal_Focus",
        "Primary_Actor_Type",
        "Geographic_Scope",
        "Relationship_Model_Definition",
        "Discourse_Type",
    ]


def _locate_unit(unit_text: str, full_text: str) -> Optional[Tuple[int, int]]:
    if not unit_text or not full_text:
        return None

    try:
        start = full_text.index(unit_text)
        return start, start + len(unit_text)
    except ValueError:
        pass

    processed_unit = utils.default_process(unit_text)
    processed_full = utils.default_process(full_text)
    alignment = fuzz.partial_ratio_alignment(processed_unit, processed_full, score_cutoff=85)
    if alignment:
        return alignment.dest_start, alignment.dest_end
    return None


def _get_text(row: pd.Series, preferred: str) -> str:
    if preferred in row and pd.notna(row[preferred]):
        text = str(row[preferred]).strip()
        if text and text.lower() != "nan":
            return text
    for col in _UNIT_TEXT_CANDIDATES:
        if col in row and pd.notna(row[col]):
            text = str(row[col]).strip()
            if text and text.lower() != "nan":
                return text
    return ""


def _get_id(row: pd.Series, preferred: str) -> str:
    if preferred in row and pd.notna(row[preferred]):
        val = str(row[preferred]).strip()
        if val:
            return val
    for col in _ID_CANDIDATES:
        if col in row and pd.notna(row[col]):
            val = str(row[col]).strip()
            if val:
                return val
    return ""


# ---------------------------------------------------------------------------
# 抽象策略
# ---------------------------------------------------------------------------


class BaseSampler(ABC):
    """所有抽样器的统一接口"""

    def __init__(self, sampling_config: Dict[str, Dict], random_seed: int = 42):
        self.sampling_config = sampling_config or {}
        self.random_seed = random_seed

    @abstractmethod
    def sample(self, df_results: pd.DataFrame, df_input: pd.DataFrame) -> List[dict]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 具体抽样策略
# ---------------------------------------------------------------------------


class PrecisionSampler(BaseSampler):
    """正向（准确率）抽样"""

    def sample(self, df_results: pd.DataFrame, df_input: pd.DataFrame) -> List[dict]:
        samples: List[dict] = []
        
        # 区分媒体文本和社交媒体
        SOCIAL_MEDIA_SOURCES = {'vk', '知乎', 'zhihu'}

        # 优化：只处理实际存在的信源，避免无用日志
        if "Source" not in df_results.columns:
            UX.warn("结果数据中缺少Source列，无法进行正向抽样")
            return samples
        if "processing_status" not in df_results.columns:
            UX.warn("结果数据中缺少processing_status列，无法进行正向抽样")
            return samples
        
        actual_sources = set(df_results["Source"].dropna().unique())
        
        for source, config in self.sampling_config.items():
            precision = config.get("precision", 0)
            if precision <= 0:
                continue
            
            # 跳过不存在的信源（无日志）
            if source not in actual_sources:
                continue
                
            source_results = df_results[
                (df_results["Source"] == source)
                & (df_results["processing_status"] == ProcessingStatus.SUCCESS)
            ]
            if source_results.empty:
                UX.info(f"{source} 没有成功处理的记录，跳过正向抽样")
                continue

            take = min(len(source_results), precision)
            sampled = source_results.sample(n=take, random_state=self.random_seed)

            for _, row in sampled.iterrows():
                unit_text = _get_text(row, "Unit_Text")
                
                # 社交媒体：不需要高亮，直接使用Unit_Text
                if source in SOCIAL_MEDIA_SOURCES:
                    # VK需要附加post_text上下文
                    display_text = unit_text
                    if source == 'vk':
                        post_text = row.get("post_text", "") or row.get("Post_Text", "")
                        if post_text:
                            display_text = f"【帖子原文】\n{post_text}\n\n【评论内容】\n{unit_text}"
                    
                    record = {col: row.get(col, "") for col in _analysis_columns()}
                    record.update(
                        {
                            "Unit_ID": row.get("Unit_ID", ""),
                            "Source": source,
                            "Unit_Text": display_text,
                            "Inspector_Is_Relevant": "",
                            "Inspector_Comments": "",
                        }
                    )
                    for col in _analysis_columns():
                        record[f"Inspector_{col}"] = ""
                    samples.append(record)
                    continue
                
                # 媒体文本：需要高亮
                # 使用序号列作为文章ID（会自动fallback到_ID_CANDIDATES）
                article_id = _get_id(row, "序号")
                if not article_id:
                    continue

                # 修复：使用字符串比较，确保ID类型一致
                article_id_str = str(article_id)
                matching = df_input[
                    df_input.apply(lambda r: str(_get_id(r, "序号")), axis=1)
                    == article_id_str
                ]
                if matching.empty:
                    UX.warn(f"正向抽样：单元 {row.get('Unit_ID', 'unknown')} 的原文章未找到")
                    continue

                full_text = _get_text(matching.iloc[0], "text")
                location = _locate_unit(unit_text, full_text)
                if location:
                    start, end = location
                    highlighted = (
                        full_text[:start]
                        + "\n【🌟===高亮段落开始===🌟】\n"
                        + full_text[start:end]
                        + "\n【🌟===高亮段落结束===🌟】\n"
                        + full_text[end:]
                    )
                else:
                    highlighted = f"【定位失败】{unit_text}"

                record = {col: row.get(col, "") for col in _analysis_columns()}
                record.update(
                    {
                        "Unit_ID": row.get("Unit_ID", ""),
                        "Article_ID": article_id,
                        "Source": source,
                        "Unit_Text": unit_text,
                        "Highlighted_Full_Text": highlighted,
                        "Inspector_Is_Relevant": "",
                        "Inspector_Boundary_Quality": "",
                        "Inspector_Comments": "",
                    }
                )
                for col in _analysis_columns():
                    record[f"Inspector_{col}"] = ""

                samples.append(record)

        return samples


class RecallByExclusionSampler(BaseSampler):
    """媒体文本：挖除版召回率"""

    def sample(self, df_results: pd.DataFrame, df_input: pd.DataFrame) -> List[dict]:
        samples: List[dict] = []
        
        # 只处理媒体文本类型的source，不处理社交媒体
        SOCIAL_MEDIA_SOURCES = {'vk', '知乎', 'zhihu'}

        # 优化：只处理实际存在的信源，避免无用日志
        if "Source" not in df_input.columns:
            UX.warn("输入数据中缺少Source列，无法进行反向抽样")
            return samples
        
        actual_sources = set(df_input["Source"].dropna().unique())

        for source, config in self.sampling_config.items():
            recall = config.get("recall", 0)
            if recall <= 0:
                continue
            
            # 跳过社交媒体source（它们由RecallByRejectionSampler处理）
            if source in SOCIAL_MEDIA_SOURCES:
                continue
            
            # 跳过不存在的信源（无日志）
            if source not in actual_sources:
                continue
                
            source_inputs = df_input[df_input["Source"] == source]
            if source_inputs.empty:
                UX.info(f"{source} 在输入数据中没有记录，跳过反向抽样")
                continue

            take = min(len(source_inputs), recall)
            sampled = source_inputs.sample(n=take, random_state=self.random_seed)

            for _, article in sampled.iterrows():
                # 使用序号列作为文章ID
                article_id = _get_id(article, "序号")
                full_text = _get_text(article, "text")
                if not full_text:
                    continue

                # 修复：使用字符串比较，确保ID类型一致
                article_id_str = str(article_id)
                units = df_results[
                    df_results.apply(lambda r: str(_get_id(r, "序号")), axis=1) == article_id_str
                ]
                
                if units.empty:
                    UX.info(f"文章 {article_id} 没有找到对应的提取单元，可能全部被判定为不相关")
                
                unit_texts = [_get_text(r, "Unit_Text") for _, r in units.iterrows() if _get_text(r, "Unit_Text")]

                positions: List[Tuple[int, int]] = []
                failed_units: List[str] = []  # 记录定位失败的单元
                
                for unit_text in unit_texts:
                    location = _locate_unit(unit_text, full_text)
                    if location:
                        positions.append(location)
                    else:
                        # 记录定位失败的单元（截取前80字符作为标识）
                        preview = unit_text[:80] + "..." if len(unit_text) > 80 else unit_text
                        failed_units.append(preview)

                positions.sort(key=lambda x: x[0], reverse=True)
                modified = full_text
                for start, end in positions:
                    modified = modified[:start] + "【已提取】" + modified[end:]

                # 如果有定位失败的单元，在末尾添加说明
                if failed_units:
                    modified += "\n\n" + "="*50 + "\n"
                    modified += "⚠️ 以下单元已被AI提取，但在原文中定位失败（可能因文本细微差异）：\n"
                    modified += "（这些内容已被提取，非遗漏。请检查AI的Unit_Text列。）\n"
                    modified += "="*50 + "\n"
                    for i, unit in enumerate(failed_units, 1):
                        modified += f"\n{i}. {unit}\n"

                if modified.replace("【已提取】", "").strip():
                    samples.append(
                        {
                            "Article_ID": article_id,
                            "Source": source,
                            "Extracted_Units_Count": len(positions),
                            "Failed_Locate_Count": len(failed_units),  # 新增：记录定位失败数量
                            "Remaining_Text": modified,
                            "Inspector_Has_Missed_Content": "",
                            "Inspector_Missed_Content_Type": "",
                            "Inspector_Comments": "",
                        }
                    )

        return samples


class RecallByRejectionSampler(BaseSampler):
    """社交媒体：抽样被判不相关的单元（仅处理有相关性判断的社交媒体）"""

    def sample(self, df_results: pd.DataFrame, df_input: pd.DataFrame) -> List[dict]:
        samples: List[dict] = []
        
        # 只处理有相关性判断的社交媒体（VK）
        # 知乎：使用ZHIHU_CHUNKING，没有相关性判断，不会有NO_RELEVANT
        # 媒体文本：NO_RELEVANT极少，由RecallByExclusionSampler统一处理，或人工筛选
        SOCIAL_MEDIA_WITH_RELEVANCE_CHECK = {'vk'}

        # 优化：只处理实际存在的信源，避免无用日志
        if "Source" not in df_results.columns:
            UX.warn("结果数据中缺少Source列，无法进行召回率抽样")
            return samples
        if "processing_status" not in df_results.columns:
            UX.warn("结果数据中缺少processing_status列，无法进行召回率抽样")
            return samples
        
        actual_sources = set(df_results["Source"].dropna().unique())

        for source, config in self.sampling_config.items():
            recall = config.get("recall", 0)
            if recall <= 0:
                continue
            
            # 跳过非社交媒体信源
            if source not in SOCIAL_MEDIA_WITH_RELEVANCE_CHECK:
                continue
            
            # 跳过不存在的信源（无日志）
            if source not in actual_sources:
                continue
                
            rejected = df_results[
                (df_results["Source"] == source)
                & (df_results["processing_status"] == ProcessingStatus.NO_RELEVANT)
            ]
            if rejected.empty:
                UX.info(f"{source} 没有被拒绝的记录，跳过召回率抽样")
                continue

            take = min(len(rejected), recall)
            sampled = rejected.sample(n=take, random_state=self.random_seed)

            for _, row in sampled.iterrows():
                unit_text = _get_text(row, "Unit_Text")
                
                # VK需要附加post_text上下文
                if source == 'vk':
                    post_text = row.get("post_text", "") or row.get("Post_Text", "")
                    if post_text:
                        unit_text = f"【帖子原文】\n{post_text}\n\n【评论内容】\n{unit_text}"
                
                samples.append(
                    {
                        "Unit_ID": row.get("Unit_ID", ""),
                        "Source": source,
                        "Unit_Text": unit_text,
                        "Inspector_Has_Missed_Content": "",
                        "Inspector_Missed_Content_Type": "",
                        "Inspector_Comments": "",
                    }
                )

        return samples


# ---------------------------------------------------------------------------
# 协调器
# ---------------------------------------------------------------------------


class ReliabilityTestModule:
    def __init__(self, output_path: str, sampling_config: dict, random_seed: int = 42):
        self.output_path = output_path
        self.sampling_config = sampling_config or {}
        self.random_seed = random_seed
        self._locale_cache: Dict[str, Dict[str, str]] = {}

        self.precision_sampler = PrecisionSampler(self.sampling_config, random_seed)
        self.recall_exclusion_sampler = RecallByExclusionSampler(self.sampling_config, random_seed)
        self.recall_rejection_sampler = RecallByRejectionSampler(self.sampling_config, random_seed)

    def generate_files(self, df_results: pd.DataFrame, df_input: pd.DataFrame) -> None:
        UX.info("开始生成信度检验文件")

        precision_samples = self.precision_sampler.sample(df_results, df_input)
        recall_exclusion_samples = self.recall_exclusion_sampler.sample(df_results, df_input)
        recall_rejection_samples = self.recall_rejection_sampler.sample(df_results, df_input)

        if precision_samples:
            self._save_positive(precision_samples)
        else:
            UX.warn("正向检验未抽到样本")

        negative_samples = recall_exclusion_samples + recall_rejection_samples
        if negative_samples:
            self._save_negative(negative_samples)
        else:
            UX.warn("反向检验未抽到样本")

        UX.ok("信度检验文件生成完成")

    # ------------------------------------------------------------------
    # 保存逻辑
    # ------------------------------------------------------------------

    def _save_positive(self, samples: List[dict]) -> None:
        df_output = pd.DataFrame(samples)
        order = ["Unit_ID", "Article_ID", "Source", "Unit_Text", "Highlighted_Full_Text"]
        for col in _analysis_columns():
            if col in df_output.columns:
                order.extend([col, f"Inspector_{col}"])
        order.extend(["Inspector_Is_Relevant", "Inspector_Boundary_Quality", "Inspector_Comments"])
        order = [col for col in order if col in df_output.columns]
        df_output = df_output.reindex(columns=order)

        zh_path = os.path.join(self.output_path, "正向检验_高亮版.xlsx")
        ru_path = os.path.join(self.output_path, "Проверка_положительная.xlsx")
        self._save_bilingual(df_output, zh_path, ru_path)
        UX.ok(f"正向检验文件已保存：{len(df_output)} 条样本")

    def _save_negative(self, samples: List[dict]) -> None:
        df_output = pd.DataFrame(samples)
        
        # 定义反向检验的列顺序（包含媒体文本和社交媒体两种数据结构）
        order = [
            # ID列（有的数据有Unit_ID，有的有Article_ID）
            "Unit_ID", 
            "Article_ID", 
            "Source",
            # 内容列
            "Unit_Text",           # VK被拒绝评论使用
            "Remaining_Text",      # 媒体文本挖除法使用
            # 统计列（仅媒体文本有）
            "Extracted_Units_Count",
            "Failed_Locate_Count",
            # 检查员列
            "Inspector_Has_Missed_Content",
            "Inspector_Missed_Content_Type",
            "Inspector_Comments"
        ]
        # 只保留实际存在的列，避免报错
        order = [col for col in order if col in df_output.columns]
        df_output = df_output.reindex(columns=order)
        
        zh_path = os.path.join(self.output_path, "反向检验.xlsx")
        ru_path = os.path.join(self.output_path, "Проверка_отрицательная.xlsx")
        self._save_bilingual(df_output, zh_path, ru_path)
        UX.ok(f"反向检验文件已保存：{len(df_output)} 条样本")

    # ------------------------------------------------------------------
    # 双语导出
    # ------------------------------------------------------------------

    def _save_bilingual(self, df: pd.DataFrame, zh_path: str, ru_path: str) -> None:
        self._ensure_dirs([zh_path, ru_path])
        try:
            df_zh = self._decorate_headers(df, "zh")
            df_zh.to_excel(zh_path, index=False)
        except Exception as e:  # pragma: no cover
            UX.warn(f"中文导出失败: {e}")
        try:
            df_ru = self._decorate_headers(df, "ru")
            df_ru.to_excel(ru_path, index=False)
        except Exception as e:  # pragma: no cover
            UX.warn(f"俄文导出失败: {e}")

    def _decorate_headers(self, df: pd.DataFrame, lang: str) -> pd.DataFrame:
        mapping = self._load_locale(lang)
        new_cols = [f"{mapping.get(col, col)}({col})" if col in mapping else col for col in df.columns]
        df_out = df.copy()
        df_out.columns = new_cols
        return df_out

    def _load_locale(self, lang: str) -> Dict[str, str]:
        if lang in self._locale_cache:
            return self._locale_cache[lang]
        locales_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "locales")
        path = os.path.join(locales_dir, f"{lang}.json")
        try:
            with open(path, "r", encoding="utf-8") as f:
                mapping = json.load(f)
                self._locale_cache[lang] = mapping
                return mapping
        except Exception:  # pragma: no cover
            UX.warn(f"本地化文件缺失: {path}")
            return {}

    @staticmethod
    def _ensure_dirs(paths: List[str]) -> None:
        for path in paths:
            directory = os.path.dirname(path)
            if directory:
                os.makedirs(directory, exist_ok=True)


def create_reliability_test_module(output_path: str, sampling_config: dict, random_seed: int = 42) -> ReliabilityTestModule:
    """创建信度检验模块实例"""
    return ReliabilityTestModule(output_path, sampling_config, random_seed)
