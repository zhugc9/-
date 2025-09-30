# -*- coding: utf-8 -*-
"""信度检验文件生成模块"""

import os
import glob
import pandas as pd
from typing import Tuple, Optional
import re
import json
try:
    from rapidfuzz import fuzz, process, utils
except ImportError:
    # 降级到thefuzz作为备选
    from thefuzz import fuzz, process
    from thefuzz import utils

# 从utils模块导入ProcessingStatus
from .utils import ProcessingStatus

class ReliabilityTestModule:
    
    def __init__(self, input_path: str, output_path: str, sampling_config: dict, 
                 id_column: str = "序号", text_column: str = "text", random_seed: int = 42):
        self.input_path = input_path
        self.output_path = output_path
        self.sampling_config = sampling_config
        self.id_column = id_column
        self.text_column = text_column
        self.random_seed = random_seed
        self._locale_cache = {}
        
    def locate_unit(self, unit_text: str, full_text: str) -> Optional[Tuple[int, int]]:
        """
        【SOTA 最终版】文本定位算法
        使用 fuzz.partial_ratio_alignment 实现精确、高效的模糊定位
        """
        if not unit_text or not full_text:
            return None
            
        # 第一级：精确匹配 (最快，保持不变)
        try:
            start_idx = full_text.index(unit_text)
            return (start_idx, start_idx + len(unit_text))
        except ValueError:
            pass  # 精确匹配失败，进入下一级
        
        # 第二级：SOTA 模糊定位 (使用 alignment)
        # 准备文本：对于字母语言，统一小写；对于CJK，可以不处理，但为了统一，这里也处理
        # utils.default_process 会智能处理，比 .lower() 更健壮
        processed_unit = utils.default_process(unit_text)
        processed_full = utils.default_process(full_text)
        
        # 核心：使用 partial_ratio_alignment 直接获取位置和分数
        # score_cutoff 的值可以根据经验调整，75-85 通常是比较合理的范围
        alignment = fuzz.partial_ratio_alignment(
            processed_unit, 
            processed_full, 
            score_cutoff=85
        )
        
        if alignment:
            # alignment 对象直接包含了我们需要的在 full_text 中的坐标
            # alignment.dest_start, alignment.dest_end
            return (alignment.dest_start, alignment.dest_end)
        
        # 如果两级都失败，则返回 None
        return None
    
    
    def generate_positive_test_file(self, df_all_results: pd.DataFrame, df_all_input: pd.DataFrame) -> None:
        print("🎯 开始生成正向检验文件（高亮版）...")
        sampled_units = []
        
        for source, config in self.sampling_config.items():
            precision_count = config.get('precision', 0)
            if precision_count <= 0:
                continue
                
            source_results = df_all_results[
                (df_all_results.get('Source', '') == source) & 
                (df_all_results.get('processing_status', '') == ProcessingStatus.SUCCESS)
            ]
            if len(source_results) == 0:
                continue
            sample_size = min(precision_count, len(source_results))
            sampled = source_results.sample(n=sample_size, random_state=self.random_seed)
            sampled_units.append(sampled)
        
        if not sampled_units:
            print("没有找到可用于正向检验的样本")
            return
            
        df_sampled = pd.concat(sampled_units, ignore_index=True)
        results = []
        for _, row in df_sampled.iterrows():
            unit_text = str(row.get('Unit_Text', ''))
            article_id = str(row.get(self.id_column, ''))
            
            full_text = ""
            matching_input = df_all_input[df_all_input[self.id_column].astype(str) == article_id]
            if not matching_input.empty:
                full_text = str(matching_input.iloc[0][self.text_column])
            
            location = self.locate_unit(unit_text, full_text)
            if location and full_text:
                start, end = location
                highlight_head = "\n【🌟高亮段落开始🌟】\n"
                highlight_tail = "\n【🌟高亮段落结束🌟】\n"
                highlighted_text = (
                    full_text[:start] +
                    highlight_head +
                    full_text[start:end] +
                    highlight_tail +
                    full_text[end:]
                )
            else:
                highlighted_text = f"【定位失败】{unit_text}"
            # 步骤一：先将原始分析结果的所有相关列复制过来
            # 创建一个包含所有分析列的列表
            analysis_columns = [
                "Incident", "Frame_SolutionRecommendation", "Frame_ResponsibilityAttribution",
                "Frame_CausalExplanation", "Frame_MoralEvaluation", "Frame_ProblemDefinition",
                "Frame_ActionStatement", "Valence", "Evidence_Type", "Attribution_Level",
                "Temporal_Focus", "Primary_Actor_Type", "Geographic_Scope",
                "Relationship_Model_Definition", "Discourse_Type"
            ]

            # 从原始行(row)中提取所有分析列的数据，存入一个新字典
            result_record = {}
            for col in analysis_columns:
                if col in row:
                    result_record[col] = row[col]

            # 步骤二：再添加和更新检验文件特有的列
            result_record.update({
                'Unit_ID': row.get('Unit_ID', ''),
                'Article_ID': article_id,
                'Source': row.get('Source', ''),
                'Unit_Text': unit_text,
                'Highlighted_Full_Text': highlighted_text,
                'Inspector_Is_Relevant': '',          # 检验员填写的列
                'Inspector_Boundary_Quality': '',     # 检验员填写的列
                'Inspector_Comments': ''              # 检验员填写的列
            })

            # 【新增逻辑开始】
            # 再次定义需要被检验的AI分析列
            analysis_columns_to_inspect = [
                "Incident", "Valence", "Evidence_Type", "Attribution_Level",
                "Temporal_Focus", "Primary_Actor_Type", "Geographic_Scope",
                "Relationship_Model_Definition", "Discourse_Type",
                "Frame_SolutionRecommendation", "Frame_ResponsibilityAttribution",
                "Frame_CausalExplanation", "Frame_MoralEvaluation", "Frame_ProblemDefinition",
                "Frame_ActionStatement"
            ]

            # 为每一个需要检验的列，动态添加一个对应的 `Inspector_` 空白列
            inspector_fields = {}
            for col in analysis_columns_to_inspect:
                inspector_fields[f"Inspector_{col}"] = ''

            # 将空白检验列更新到记录中
            result_record.update(inspector_fields)
            # 【新增逻辑结束】

            # 将此更新后的result_record添加到results列表中
            results.append(result_record)
        
        # 保存文件
        if results:
            df_output = pd.DataFrame(results)

            # 【新增逻辑开始】
            # 定义最终的列顺序
            final_column_order = [
                'Unit_ID', 'Article_ID', 'Source', 'Unit_Text', 'Highlighted_Full_Text'
            ]

            # 定义需要被检验的AI分析列（与上面保持一致）
            analysis_columns_to_inspect = [
                "Incident", "Valence", "Evidence_Type", "Attribution_Level",
                "Temporal_Focus", "Primary_Actor_Type", "Geographic_Scope",
                "Relationship_Model_Definition", "Discourse_Type",
                "Frame_SolutionRecommendation", "Frame_ResponsibilityAttribution",
                "Frame_CausalExplanation", "Frame_MoralEvaluation", "Frame_ProblemDefinition",
                "Frame_ActionStatement"
            ]

            # 动态构建"AI分析列"和"检验列"并排的顺序
            for col in analysis_columns_to_inspect:
                if col in df_output.columns:
                    final_column_order.append(col)
                    final_column_order.append(f"Inspector_{col}")

            # 添加遗留的检验员评论列
            final_column_order.extend(['Inspector_Is_Relevant', 'Inspector_Boundary_Quality', 'Inspector_Comments'])

            # 过滤掉不存在于DataFrame中的列名，并重排
            existing_columns_ordered = [col for col in final_column_order if col in df_output.columns]
            df_output = df_output.reindex(columns=existing_columns_ordered)
            # 【新增逻辑结束】

            zh_positive_path = os.path.join(self.output_path, '正向检验及框架维度检验_高亮版.xlsx')
            ru_positive_path = os.path.join(self.output_path, 'Проверка_положительная_выборка(точность_и_границы).xlsx')
            self._save_bilingual(df_output, zh_positive_path, ru_positive_path)
            print(f"正向检验及框架维度检验文件已生成: {zh_positive_path}")
            print(f"   样本数量: {len(results)}")
        else:
            print("没有生成正向检验样本")
    
    def generate_negative_test_file(self, df_all_results: pd.DataFrame, df_all_input: pd.DataFrame) -> None:
        print("🔍 开始生成反向检验文件（挖除版）...")
        
        # 按来源抽样文章
        sampled_articles = []
        for source, config in self.sampling_config.items():
            recall_count = config.get('recall', 0)
            if recall_count <= 0:
                continue
            
            source_input = df_all_input[df_all_input.get('Source', '') == source]
            if len(source_input) == 0:
                continue
                
            sample_size = min(recall_count, len(source_input))
            sampled = source_input.sample(n=sample_size, random_state=self.random_seed)
            sampled_articles.append(sampled)
        
        if not sampled_articles:
            print("没有找到可用于反向检验的文章")
            return
            
        df_sampled_articles = pd.concat(sampled_articles, ignore_index=True)
        results = []
        
        for _, article_row in df_sampled_articles.iterrows():
            article_id_str = str(article_row[self.id_column])
            full_text = str(article_row[self.text_column])
            
            # 获取该文章的所有议题单元
            article_units = df_all_results[df_all_results[self.id_column].astype(str) == article_id_str]
            unit_texts = article_units['Unit_Text'].dropna().tolist() if not article_units.empty else []
            
            # 收集所有有效的位置信息
            positions = []
            for unit_text in unit_texts:
                location = self.locate_unit(str(unit_text), full_text)
                if location:
                    positions.append(location)
            
            # 按start_index降序排序（从大到小）
            positions.sort(key=lambda x: x[0], reverse=True)
            
            # 创建full_text的可变副本并执行替换
            modified_text = full_text
            for start, end in positions:
                modified_text = modified_text[:start] + "【已提取】" + modified_text[end:]
            
            # 检查挖除后是否还有剩余的有效文本
            check_text = modified_text.replace("【已提取】", "").strip()
            
            # 只有当剩余文本不为空时，才将该记录添加到结果中
            if check_text:
                result_record = {
                    'Article_ID': article_id_str,
                    'Source': article_row.get('Source', ''),
                    'Extracted_Units_Count': len(positions),
                    'Remaining_Text': modified_text,
                    'Inspector_Has_Missed_Content': '',
                    'Inspector_Missed_Content_Type': '',
                    'Inspector_Comments': ''
                }
                results.append(result_record)
        
        # 保存文件
        if results:
            df_output = pd.DataFrame(results)
            zh_negative_path = os.path.join(self.output_path, '反向检验_挖除版.xlsx')
            ru_negative_path = os.path.join(self.output_path, 'Проверка_исключенный_текст(полнота).xlsx')
            self._save_bilingual(df_output, zh_negative_path, ru_negative_path)
            print(f"反向检验文件已生成: {zh_negative_path}")
            print(f"   文章数量: {len(results)}")
        else:
            print("没有生成反向检验样本")
    
    def generate_reliability_files(self) -> None:
        print("启动信度检验文件生成模块（SOTA流式处理版）")
        print("=" * 60)
        
        # 【内存优化】第一步：只获取文件路径，不加载内容
        analyzed_pattern = os.path.join(self.output_path, "*analyzed_*.xlsx")
        analyzed_files = glob.glob(analyzed_pattern)
        
        if not analyzed_files:
            print(f"未找到分析结果文件，搜索模式: {analyzed_pattern}")
            return
            
        print(f"找到 {len(analyzed_files)} 个分析结果文件:")
        for file in analyzed_files:
            print(f"   - {os.path.basename(file)}")
        
        input_files = []
        if os.path.isdir(self.input_path):
            input_pattern = os.path.join(self.input_path, "*.xlsx")
            input_files = glob.glob(input_pattern)
        elif os.path.isfile(self.input_path) and self.input_path.lower().endswith('.xlsx'):
            input_files = [self.input_path]
        else:
            print(f"输入路径无效: {self.input_path}")
        
        temp_input_dir = os.path.join(self.output_path, "_reliability_inputs")
        if os.path.isdir(temp_input_dir):
            temp_files = glob.glob(os.path.join(temp_input_dir, "*.xlsx"))
            input_files.extend(temp_files)

        if not input_files:
            print(f"未找到原始输入文件，搜索路径: {self.input_path}")
            return
            
        print(f"找到 {len(input_files)} 个原始输入文件:")
        for file in input_files:
            print(f"   - {os.path.basename(file)}")
        
        # 【内存优化】第二步：逐文件流式处理
        print(f"\n采用流式处理模式，逐文件处理以节省内存")
        os.makedirs(self.output_path, exist_ok=True)
        
        # 收集样本数据的容器（只保存抽样结果，不是全量数据）
        positive_samples = []
        negative_samples = []
        
        # 主循环：逐个处理输入文件
        for input_file in input_files:
            print(f"\n正在处理: {os.path.basename(input_file)}")
            
            try:
                # 只加载当前一个输入文件
                df_input_single = pd.read_excel(input_file)
                if not self._validate_input_columns(df_input_single):
                    print(f"   跳过文件（缺少必要列）: {os.path.basename(input_file)}")
                    continue
                
                input_source = self._identify_source(os.path.basename(input_file))
                df_input_single['Source'] = input_source
                print(f"   输入文件加载成功: {len(df_input_single)} 条记录")
                
                # 找到对应的分析结果文件
                corresponding_analyzed = None
                for analyzed_file in analyzed_files:
                    if self._files_match(input_file, analyzed_file):
                        corresponding_analyzed = analyzed_file
                        break
                
                if not corresponding_analyzed:
                    print(f"   未找到对应的分析结果文件，跳过")
                    continue
                
                # 只加载对应的分析结果文件
                df_analyzed_single = pd.read_excel(corresponding_analyzed)
                analyzed_source = self._identify_source(os.path.basename(corresponding_analyzed))
                df_analyzed_single['Source'] = analyzed_source
                print(f"   分析结果加载成功: {len(df_analyzed_single)} 条记录")
                
                # 生成当前文件的样本（返回抽样数据，不直接写文件）
                pos_samples = self._generate_positive_samples(df_analyzed_single, df_input_single)
                neg_samples = self._generate_negative_samples(df_analyzed_single, df_input_single)
                
                positive_samples.extend(pos_samples)
                negative_samples.extend(neg_samples)
                print(f"   本文件贡献: 正向样本{len(pos_samples)}个, 反向样本{len(neg_samples)}个")
                
            except Exception as e:
                print(f"   处理文件失败: {e}")
                continue
        
        # 【内存优化】第三步：最后统一保存抽样结果
        print(f"\n正在保存最终的信度检验文件...")
        
        try:
            if positive_samples:
                self._save_positive_test_file(positive_samples)
                print(f"正向检验文件已生成，共 {len(positive_samples)} 个样本")
            else:
                print("没有正向检验样本")
        except Exception as e:
            print(f"正向检验文件生成失败: {e}")
        
        try:
            if negative_samples:
                self._save_negative_test_file(negative_samples)
                print(f"反向检验文件已生成，共 {len(negative_samples)} 个样本")
            else:
                print("没有反向检验样本")
        except Exception as e:
            print(f"反向检验文件生成失败: {e}")
        
        print("\n" + "=" * 60)
        print("流式信度检验文件生成完成！")
    
    def _validate_input_columns(self, df_input: pd.DataFrame) -> bool:
        if df_input is None or df_input.empty:
            return False

        columns = set(df_input.columns)
        required_pairs = [
            (self.id_column, self.text_column),      # 默认配置
            ('序号', 'text'),                        # 媒体文本原始列
            ('comment_id', 'comment_text'),          # VK 原始列
            ('id', '回答内容'),                       # 知乎原始列
            ('序号', '回答内容')                      # 知乎 Excel 另一种常见命名
        ]

        for id_col, text_col in required_pairs:
            if id_col in columns and text_col in columns:
                return True

        print(f"   输入列不满足要求，现有列: {list(columns)}")
        return False

    def _files_match(self, input_file: str, analyzed_file: str) -> bool:
        """判断输入文件和分析结果文件是否匹配"""
        input_basename = os.path.basename(input_file).replace('.xlsx', '')
        analyzed_basename = os.path.basename(analyzed_file).replace('(不能删)analyzed_', '').replace('.xlsx', '')
        return input_basename == analyzed_basename
    
    def _generate_positive_samples(self, df_analyzed: pd.DataFrame, df_input: pd.DataFrame) -> list:
        """生成正向检验样本（基于原generate_positive_test_file逻辑）"""
        samples = []
        source = df_analyzed.get('Source', '').iloc[0] if not df_analyzed.empty else '未知来源'
        
        if source not in self.sampling_config:
            return samples
            
        precision_count = self.sampling_config[source].get('precision', 0)
        if precision_count <= 0:
            return samples
        
        # 筛选成功记录
        source_results = df_analyzed[
            (df_analyzed.get('Source', '') == source) & 
            (df_analyzed.get('processing_status', '') == ProcessingStatus.SUCCESS)
        ]
        
        if len(source_results) == 0:
            return samples
            
        # 抽样
        sample_size = min(precision_count, len(source_results))
        sampled = source_results.sample(n=sample_size, random_state=self.random_seed)
        
        # 生成样本记录
        for _, row in sampled.iterrows():
            unit_text = str(row.get('Unit_Text', ''))
            article_id = str(row.get(self.id_column, ''))
            
            # 获取完整原文
            full_text = ""
            matching_input = df_input[df_input[self.id_column].astype(str) == article_id]
            if not matching_input.empty:
                full_text = str(matching_input.iloc[0][self.text_column])
            
            # 文本高亮定位
            location = self.locate_unit(unit_text, full_text)
            if location and full_text:
                start, end = location
                highlighted_text = (
                    full_text[:start] + 
                    "【" + full_text[start:end] + "】" + 
                    full_text[end:]
                )
            else:
                highlighted_text = f"【定位失败】{unit_text}"
            
            # 构建样本记录（包含AI分析结果和检验员空白列）
            sample_record = self._build_positive_sample_record(row, article_id, source, unit_text, highlighted_text)
            samples.append(sample_record)
        
        return samples
    
    def _generate_negative_samples(self, df_analyzed: pd.DataFrame, df_input: pd.DataFrame) -> list:
        """生成反向检验样本（基于原generate_negative_test_file逻辑）"""
        samples = []
        source = df_analyzed.get('Source', '').iloc[0] if not df_analyzed.empty else '未知来源'
        
        if source not in self.sampling_config:
            return samples
            
        recall_count = self.sampling_config[source].get('recall', 0)
        if recall_count <= 0:
            return samples
        
        # 从输入文件中抽样文章
        source_input = df_input[df_input.get('Source', '') == source]
        if len(source_input) == 0:
            return samples
            
        sample_size = min(recall_count, len(source_input))
        sampled_articles = source_input.sample(n=sample_size, random_state=self.random_seed)
        
        # 处理每篇被抽样的文章
        for _, article_row in sampled_articles.iterrows():
            article_id_str = str(article_row[self.id_column])
            full_text = str(article_row[self.text_column])
            
            # 获取该文章的所有议题单元
            article_units = df_analyzed[df_analyzed[self.id_column].astype(str) == article_id_str]
            unit_texts = article_units['Unit_Text'].dropna().tolist() if not article_units.empty else []
            
            # 执行挖除操作
            positions = []
            for unit_text in unit_texts:
                location = self.locate_unit(str(unit_text), full_text)
                if location:
                    positions.append(location)
            
            # 按start_index降序排序并挖除
            positions.sort(key=lambda x: x[0], reverse=True)
            modified_text = full_text
            for start, end in positions:
                modified_text = modified_text[:start] + "【已提取】" + modified_text[end:]
            
            # 检查是否还有剩余有效文本
            check_text = modified_text.replace("【已提取】", "").strip()
            if check_text:
                sample_record = {
                    'Article_ID': article_id_str,
                    'Source': article_row.get('Source', ''),
                    'Extracted_Units_Count': len(positions),
                    'Remaining_Text': modified_text,
                    'Inspector_Has_Missed_Content': '',
                    'Inspector_Missed_Content_Type': '',
                    'Inspector_Comments': ''
                }
                samples.append(sample_record)
        
        return samples

    def _identify_source(self, filename: str) -> str:
        """根据文件名识别来源，与主脚本逻辑保持一致"""
        source_map = {
            '俄总统': ['俄总统', '总统', 'Putin', 'president'],
            '俄语媒体': ['俄语媒体', '俄语', 'russian', 'ru_media', '俄媒'],
            '中文媒体': ['中文媒体', '中文', 'chinese', 'cn_media', '中媒', '新华社'],
            '英语媒体': ['英语媒体', '英语', 'english', 'en_media', '英媒'],
            'vk': ['vk'],
            '知乎': ['知乎', 'zhihu']
        }

        filename_lower = filename.lower()
        for source, keywords in source_map.items():
            if any(kw.lower() in filename_lower for kw in keywords):
                return source

        return '未知来源'
    
    def _build_positive_sample_record(self, row, article_id: str, source: str, unit_text: str, highlighted_text: str) -> dict:
        """构建正向检验样本记录"""
        # 先复制AI分析结果的所有相关列
        analysis_columns = [
            "Incident", "Frame_SolutionRecommendation", "Frame_ResponsibilityAttribution",
            "Frame_CausalExplanation", "Frame_MoralEvaluation", "Frame_ProblemDefinition",
            "Frame_ActionStatement", "Valence", "Evidence_Type", "Attribution_Level",
            "Temporal_Focus", "Primary_Actor_Type", "Geographic_Scope",
            "Relationship_Model_Definition", "Discourse_Type"
        ]

        result_record = {}
        for col in analysis_columns:
            if col in row:
                result_record[col] = row[col]

        # 添加检验文件特有的列
        result_record.update({
            'Unit_ID': row.get('Unit_ID', ''),
            'Article_ID': article_id,
            'Source': source,
            'Unit_Text': unit_text,
            'Highlighted_Full_Text': highlighted_text,
            'Inspector_Is_Relevant': '',
            'Inspector_Boundary_Quality': '',
            'Inspector_Comments': ''
        })

        # 为每个AI分析列添加对应的Inspector检验列
        analysis_columns_to_inspect = [
            "Incident", "Valence", "Evidence_Type", "Attribution_Level",
            "Temporal_Focus", "Primary_Actor_Type", "Geographic_Scope",
            "Relationship_Model_Definition", "Discourse_Type",
            "Frame_SolutionRecommendation", "Frame_ResponsibilityAttribution",
            "Frame_CausalExplanation", "Frame_MoralEvaluation", "Frame_ProblemDefinition",
            "Frame_ActionStatement"
        ]

        for col in analysis_columns_to_inspect:
            result_record[f"Inspector_{col}"] = ''

        return result_record
    
    def _save_positive_test_file(self, samples: list) -> None:
        """保存正向检验文件"""
        if not samples:
            return
        
        df_output = pd.DataFrame(samples)

        # 定义最终的列顺序（AI分析列和检验列交错）
        final_column_order = [
            'Unit_ID', 'Article_ID', 'Source', 'Unit_Text', 'Highlighted_Full_Text'
        ]

        analysis_columns_to_inspect = [
            "Incident", "Valence", "Evidence_Type", "Attribution_Level",
            "Temporal_Focus", "Primary_Actor_Type", "Geographic_Scope",
            "Relationship_Model_Definition", "Discourse_Type",
            "Frame_SolutionRecommendation", "Frame_ResponsibilityAttribution",
            "Frame_CausalExplanation", "Frame_MoralEvaluation", "Frame_ProblemDefinition",
            "Frame_ActionStatement"
        ]

        # 动态构建"AI分析列"和"检验列"并排的顺序
        for col in analysis_columns_to_inspect:
            if col in df_output.columns:
                final_column_order.append(col)
                final_column_order.append(f"Inspector_{col}")

        # 添加遗留的检验员评论列
        final_column_order.extend(['Inspector_Is_Relevant', 'Inspector_Boundary_Quality', 'Inspector_Comments'])

        # 过滤掉不存在于DataFrame中的列名，并重排
        existing_columns_ordered = [col for col in final_column_order if col in df_output.columns]
        df_output = df_output.reindex(columns=existing_columns_ordered)

        zh_positive_path = os.path.join(self.output_path, '正向检验及框架维度检验_高亮版.xlsx')
        ru_positive_path = os.path.join(self.output_path, 'Проверка_положительная_выборка(точность_и_границы).xlsx')
        self._save_bilingual(df_output, zh_positive_path, ru_positive_path)
    
    def _save_negative_test_file(self, samples: list) -> None:
        """保存反向检验文件"""
        if not samples:
            return
        
        df_output = pd.DataFrame(samples)
        zh_negative_path = os.path.join(self.output_path, '反向检验_挖除版.xlsx')
        ru_negative_path = os.path.join(self.output_path, 'Проверка_исключенный_текст(полнота).xlsx')
        self._save_bilingual(df_output, zh_negative_path, ru_negative_path)
    
    def _load_locale_mapping(self, lang: str) -> dict:
        """从JSON文件加载本地化映射"""
        if lang in self._locale_cache:
            return self._locale_cache[lang]

        locales_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'locales')
        file_path = os.path.join(locales_dir, f'{lang}.json')

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                mapping = json.load(f)
                self._locale_cache[lang] = mapping
                return mapping
        except Exception as e:
            print(f"加载本地化文件 {lang}.json 失败: {e}")
            return {}

    def _decorate_headers(self, df: pd.DataFrame, lang: str) -> pd.DataFrame:
        """装饰列名为语言标签格式"""
        mapping = self._load_locale_mapping(lang)
        new_cols = []

        for c in list(df.columns):
            if c in mapping:
                new_cols.append(f"{mapping[c]}({c})")
            else:
                new_cols.append(c)

        df_out = df.copy()
        df_out.columns = new_cols
        return df_out

    def _save_bilingual(self, df: pd.DataFrame, zh_path: str, ru_path: str):
        """保存双语版本文件"""
        try:
            zh_dir = os.path.dirname(zh_path)
            ru_dir = os.path.dirname(ru_path)

            if zh_dir:
                os.makedirs(zh_dir, exist_ok=True)

            if ru_dir and ru_dir != zh_dir:
                os.makedirs(ru_dir, exist_ok=True)
        except Exception as e:
            print(f"创建输出目录失败: {e}")

        try:
            df_zh = self._decorate_headers(df, 'zh')
            df_zh.to_excel(zh_path, index=False)
        except Exception as e:
            print(f"中文版本导出失败: {e}")

        try:
            df_ru = self._decorate_headers(df, 'ru')
            df_ru.to_excel(ru_path, index=False)
        except Exception as e:
            print(f"俄语版本导出失败: {e}")

def create_reliability_test_module(input_path: str, output_path: str, sampling_config: dict,
                                 id_column: str = "序号", text_column: str = "text", 
                                 random_seed: int = 42) -> ReliabilityTestModule:
    return ReliabilityTestModule(input_path, output_path, sampling_config, id_column, text_column, random_seed)