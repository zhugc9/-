# 项目交接简报

## 重构完成状态
- 原 `core/processors.py` 已拆分为 `core/base_processor.py`、`core/vk_processor.py`、`core/zhihu_processor.py`、`core/media_text_processor.py`，旧文件删除。
- `core/__init__.py` 已同步导出，主入口改为按需导入上述模块。

## 修复的Bug
1. `ZhihuProcessor._get_author_name()` 增加默认返回值，避免出现空作者。
2. `MediaTextProcessor` 删除重复的哈希计算语句。
3. `config.yaml:176` VK的precision从0改为50，确保正向检验抽样。
4. `config.yaml:178` sampling_config的key从"zhihu"改为"知乎"，匹配identify_source返回值。
5. `reliability.py:208` RecallByExclusionSampler增加社交媒体过滤，避免用挖除逻辑处理VK/知乎。
6. `reliability.py:160` PrecisionSampler增加社交媒体分支，社交媒体无需Article_ID和高亮。

## 最新重构 (2025-10-06)
### 状态模型简化
- **删除PENDING_ANALYSIS状态**：第一阶段完成后直接标记STAGE_2_FAILED，待第二阶段分析
- **删除API_FAILED状态**：Worker异常直接标记为STAGE_2_FAILED，统一由断点续传处理
- **清理speaker字段污染**：
  - 新的失败记录不再写入 `speaker="API_CALL_FAILED"`
  - 断点续传读取旧记录时自动过滤掉 `API_CALL_FAILED`，恢复为实际发言人/信源
  - 解决了旧失败标记污染重试成功记录的问题
- 状态从5个简化为4个：SUCCESS, NO_RELEVANT, STAGE_1_FAILED, STAGE_2_FAILED
- 失败记录保留原文，`processing_status` 为唯一状态标识
- 旧的 `clean_failed_records()` 函数仅用于向后兼容清理旧数据

### 断点续传统一
- **BaseProcessor新增3个通用方法**：
  - `load_failed_units()` - 加载STAGE_2_FAILED单元（所有处理器）
  - `load_failed_stage1_ids()` - 加载STAGE_1_FAILED的ID（两阶段处理器）
  - `get_never_processed_ids()` - 获取未处理的ID（所有处理器）
- **删除重复实现**：
  - `main_media_text.py`的`build_resume_plan()`（34行）
  - `zhihu_processor.py`的`_load_zhihu_pending_units()`（11行）
  - `vk_processor.py`的`_load_vk_pending_units()`（38行）
- 三个处理器统一调用BaseProcessor方法，逻辑一致，净减少45行代码

### 文件修改清单
**已修改文件：**
1. `core/utils.py` - 删除PENDING_ANALYSIS常量，简化状态枚举
2. `core/base_processor.py` - 新增3个通用断点续传方法（48行）
3. `core/zhihu_processor.py` - 替换状态、删除重复方法（-11行）
4. `core/vk_processor.py` - 统一使用STAGE_2_FAILED状态
5. `main_media_text.py` - 删除build_resume_plan，使用通用方法（-34行，+15行）

### 断点续传行为
- **媒体文本**：STAGE_1_FAILED→重新提取 | STAGE_2_FAILED→复用单元重新分析
- **知乎**：STAGE_1_FAILED→重新切分 | STAGE_2_FAILED→复用章节重新分析
- **VK**：STAGE_2_FAILED→重新批量分析（无第一阶段）

### 日志系统优化 (2025-10-07)
- **精简冗余日志**：删除中间过程日志（批量保存、阶段保存、文章级详细日志），日志量减少90%
- **统计改为单元维度**：断点续传和文件完成总结使用单元数统计，更准确反映处理进度
- **移除所有emoji**：提升跨平台兼容性，避免Windows终端编码问题
- **优化用户提示**：VK批次失败日志明确说明失败数量，API失败信息缩短为30字符
- **保留关键信息**：进度条、错误/警告日志、断点续传计划、最终统计等重要日志完整保留

## 架构说明
### BaseProcessor 与断点续传
**Excel读写与列管理：**
- `_ensure_required_columns()`、`_upsert_dataframe()`、`_save_records_to_output()`、`_read_output_dataframe()` 
- `_write_stage_one_records()` 与 `_save_stage_two_results()` 实现阶段写入与覆盖更新
- `_compute_unit_hash()` / `_add_hash_to_record()` 保证单元去重

**断点续传通用方法：**
- `load_failed_units()` - 加载需要重新分析的单元（STAGE_2_FAILED）
- `load_failed_stage1_ids()` - 加载第一阶段失败的ID（STAGE_1_FAILED）
- `get_never_processed_ids()` - 获取从未处理过的ID
- 所有处理器复用，无需各自实现

### 各处理器特性
- `MediaTextProcessor`: 两阶段(提取单元→深度分析)，支持失败单元级重试
- `ZhihuProcessor`: 短文本直接分析，长文本两阶段，按回答ID组织断点续传
- `VKProcessor`: 按帖子ID批处理评论，异步并发，增量保存

### 导入路径
主入口使用: `from core import VKProcessor, ZhihuProcessor, MediaTextProcessor`
所有Processor类通过 `core/__init__.py` 统一导出，向后兼容。

## 已完成验证
- Python编译: 4个文件无语法错误
- 导入测试: 所有类可正常导入和实例化
- 继承关系: 3个子类正确继承BaseProcessor
- 属性继承: 子类正确继承thresholds和Units_collector
- Linter检查: 无错误无警告

## 运行入口与触发
- `main_media_text.py`: 读取 `config.media_text.paths` 配置的 Excel；运行完成后在输出目录生成 `(不能删)analyzed_*.xlsx`，再调用信度检验模块。
- `main_social_media.py`: 处理 `vk`、`知乎` 输入目录；在输出目录聚合结果并触发信度检验。
- 信度检验生成要求：`config.reliability_test.enabled` 为 true，且输出目录存在已处理的 `*analyzed_*.xlsx`。

## 配置结构
`config.yaml` 采用分层结构：
- `api` - credentials (keys, base_url), request_params, strategy (并发/重试/超时)
- `model_pools` - 统一模型配置入口，各处理器按stage_key查询
- `social_media` - paths, column_mapping (vk/zhihu), text_length_thresholds, vk_processing
- `media_text` - paths, column_mapping
- `processing` - general (buffer_limit), auto_retry (enabled, max_rounds, delay_minutes)
- `reliability_test` - enabled, sampling_config (各信源的precision/recall抽样数量)
- `required_output_columns` - 统一输出列定义，所有处理器共用
- 环境变量 `OPENAI_API_KEY`/`API_KEY` 和 `OPENAI_BASE_URL`/`API_BASE_URL` 可覆盖默认 API 配置。

## 信度检验模块
位置: `core/reliability.py` (402行)
触发: `config.yaml` reliability_test.enabled=true
生成文件: 4个 (正向检验中俄双语 + 反向检验中俄双语)

- `PrecisionSampler` (132-198行): 从SUCCESS记录抽样，验证准确率
  - 媒体文本: 高亮原文中的提取单元 (`Highlighted_Full_Text`列)
  - 社交媒体: 抽取成功判定为相关的单元
- `RecallByExclusionSampler` (201-254行): 媒体文本挖除版召回
  - 从原文挖除已提取单元，展示剩余文本 (`Remaining_Text`列)
- `RecallByRejectionSampler` (257-290行): 社交媒体排除版召回
  - 从NO_RELEVANT记录抽样，验证是否误判

### 技术实现
- 高亮定位: `_locate_unit()` (66-81行) 精确匹配→模糊匹配(85%阈值)
- 扣除逻辑: 倒序处理位置避免偏移 (236行)
- 双语输出: 基于 `locales/{zh,ru}.json` 映射列名

### 文件生成方式
- **统一合并模式**: 所有信源样本合并到同一文件，通过`Source`列区分
  - 正向检验：包含所有媒体文本（带高亮）+ 社交媒体（VK带帖子上下文）
  - 反向检验：包含媒体文本挖除版 + VK排除版（知乎recall=0不检验）
- **列名双语映射**: 格式为"翻译(原名)"，如"议题单元原文(Unit_Text)"
- **优势**: 检验员友好，可计算整体或分信源信度，灵活分组分析

### 设计原则与调用契约
- **契约设计**: `generate_files(df_results, df_input)` 要求输入DataFrame必须包含 `Source` 列
- **调用保证**: 主程序在调用前通过 `identify_source()` 添加Source列（main_social_media.py:262, main_media_text.py:473）
- **快速失败**: 列缺失时立即报错（KeyError），而非静默跳过，便于快速定位数据准备流程问题
- **不做防御**: 信任调用者遵守契约，避免过度防御导致问题隐蔽化

### 修复记录
- **(2025-10-06)** `zhihu_processor.py:378` 添加`Unit_ID`到类型转换列表，消除pandas警告
- **(2025-10-07)** `reliability.py:265-303` 修复反向检验静默失败：定位失败的单元现在会在`Remaining_Text`末尾明确标注，新增`Failed_Locate_Count`字段。85%模糊匹配阈值（考虑到AI返回文本细微差异）
- **(2025-10-07)** `reliability.py` 修复信度检验核心bug：DataFrame筛选从`.get()`改为`[]`（145-159/256-264/343-357行），ID比较统一转字符串避免类型不匹配（276-279/197-204行），反向检验添加列顺序定义（451-470行）。`main_media_text.py:504-505` 强制添加Source列。`core/auto_retry.py:110-114` 添加CancelledError处理
- **(2025-10-07)** `reliability.py:193,200,275,283` 修复ID关联：移除冗余的Article_ID字段，直接使用"序号"列关联文章（媒体文本输入已有序号列，无需重复），信度检验输出仍保留Article_ID列名以标准化
- **(2025-10-07)** 字段清理：`config.yaml:187-194` 从required_output_columns删除类型专用字段（序号/日期/标题/text/token数/seed_sentence/expansion_logic），只保留真正通用的分析字段，避免跨类型空列污染。`zhihu_processor.py:34-36` 删除冗余Answer_ID列（断点续传直接用序号），统一Unit_ID格式为"序号-Unit-编号"（与媒体文本一致）。`base_processor.py:180-184,195` 删除Answer_ID依赖。`reliability.py:36-40` 更新_ID_CANDIDATES，移除历史遗留的id和Article_ID
- **(2025-10-07)** `reliability.py:138-155,255-273,355-376` 日志优化：信度检验抽样时只处理实际存在的信源，避免遍历配置中所有信源导致大量"跳过XX信源"的无用日志
- **(2025-10-07)** 空列清理：`zhihu_processor.py:117,228,328,467` 删除冗余Answer_ID字段（已通过序号关联），`zhihu_processor.py:379` 删除expansion_logic字段（媒体文本专用，知乎无此逻辑）；`vk_processor.py:478` 删除Batch_ID字段（仅失败记录生成，正常记录为空）；`config.yaml:191` 从required_output_columns删除speaker（VK未实现该字段）。`reliability.py:465,494` 添加信度检验文件保存成功日志
- **(2025-10-07)** 列顺序管控：`base_processor.py:48-113` 新增`_reorder_columns`方法，智能识别数据源类型（媒体文本/知乎/VK）并应用对应的列顺序优先级，在所有保存点（第144、287、327行）调用。优先列按业务逻辑排序（源数据→Unit标识→Source信息→内容→状态→分析字段），未列出的列自动追加最后，不丢失任何数据
- **(2025-10-07)** **[严重bug]** `zhihu_processor.py:176` 修复阶段2永不执行：原先第159行在阶段1写入前就加载`pending_units`导致为空，修改为第176行阶段1写入后再加载，使阶段2能正确处理刚切分的单元。此bug导致所有知乎单元停留在STAGE_2_FAILED状态
- **(2025-10-07)** `zhihu_processor.py:161-171` 修复日志统计错误：删除Answer_ID后遗留的统计代码引用不存在字段导致回答数统计错误（多个None去重后计数为1），改用正确的映射字段answer_id_col。同时优化日志表述，增加单元数显示：`本次处理: X篇回答（Y个单元待分析）`
- **(2025-10-07)** `base_processor.py:232` 清理Answer_ID残留引用：移除`_write_stage_one_records`中尝试获取已删除Answer_ID字段的冗余代码，直接使用映射字段

## processing_status规范
所有处理器统一使用 `core/utils.py` 中的ProcessingStatus枚举（4个状态）:
- `SUCCESS` - 处理成功
- `NO_RELEVANT` - 无相关内容(媒体文本无议题单元/社交媒体判定不相关)
- `STAGE_1_FAILED` - 第一阶段失败(切分/提取失败)
- `STAGE_2_FAILED` - 第二阶段失败(分析失败、API异常、Worker错误)或第一阶段完成待第二阶段分析

断点续传逻辑依赖此字段筛选失败记录重试：
- `load_failed_units()` 加载所有 `STAGE_2_FAILED` 记录
- `load_failed_stage1_ids()` 加载所有 `STAGE_1_FAILED` 的ID
- `get_never_processed_ids()` 识别从未处理的ID
- 无需额外的清理步骤，失败记录自动重试

## 并发架构差异 (2025-10-07)

### 知乎处理器串行实现
**位置**: `core/zhihu_processor.py:396`
```python
for idx, unit_row in df_units.iterrows():
    result = await self.api_service.call_api_async(prompt, 'zh', stage_key)
```

**现状**:
- 知乎第二阶段使用串行for循环，每个章节顺序等待API返回
- `config.yaml`中的`max_concurrent_requests`参数不影响知乎处理速度
- 该参数仅控制媒体文本Worker池和VK批处理并发数

**对比VK并发实现** (`core/vk_processor.py:225-233`):
```python
semaphore = asyncio.Semaphore(max_concurrent)
tasks = [asyncio.create_task(run_batch(idx, task)) for idx, task in enumerate(batch_tasks)]
for task in asyncio.as_completed(tasks):
    batch_idx, batch_results = await task
```

**技术原因**:
- 知乎分离写盘（阶段1切分后立即写入`STAGE_2_FAILED`）是为保护切分结果，与串行/并发无关
- 串行实现可能为历史遗留，可改造为并发而不改变分离写盘逻辑

**改造方向**:
```python
semaphore = asyncio.Semaphore(max_concurrent)
tasks = [asyncio.create_task(analyze_unit(idx, row)) for idx, row in df_units.iterrows()]
for task in asyncio.as_completed(tasks):
    idx, result = await task
    df_units.at[idx, ...] = result
```

## 数据管线与ID字段统一 (2025-10-07)

### 问题背景
不同数据源的ID字段混乱（Article_ID、Answer_ID、序号、comment_id），导致断点续传和信度检验逻辑不清晰，输出Excel存在冗余列和空列。

### 解决方案
**ID字段标准化**：
- **媒体文本**：主ID为`序号`，断点续传和信度检验均基于`序号`关联
- **知乎**：主ID为`序号`，移除冗余`Answer_ID`列，`Unit_ID`格式统一为`{序号}-Unit-{编号}`
- **VK**：主ID为`comment_id`，`Unit_ID`格式为`VK-{comment_id}`（一评论=一单元）

**配置清理**：
- `reliability.py:36-40` 更新`_ID_CANDIDATES = ["序号", "Unit_ID", "comment_id"]`
- `config.yaml:187-215` 从`required_output_columns`移除源特定列（序号、日期、标题、text等）

**代码修改**：
- `media_text_processor.py:125,240` 移除冗余`Article_ID`字段
- `zhihu_processor.py:34-36` 移除冗余`Answer_ID`，统一`Unit_ID`格式
- `base_processor.py:180-195` 移除`Answer_ID`依赖，断点续传仅基于主ID

### 管线验证结果
| 数据源 | 主ID字段 | Unit_ID格式 | 断点续传 | 信度检验关联 |
|--------|----------|-------------|----------|--------------|
| 媒体文本 | `序号` | `{序号}-EN2208-Unit-1` | ✅ 基于序号 | ✅ 基于序号 |
| 知乎 | `序号` | `{序号}-ZH-E57-Unit-1` | ✅ 基于序号 | ✅ 基于序号 |
| VK | `comment_id` | `VK-{comment_id}` | ✅ 基于comment_id | ✅ 基于comment_id |

### Excel合并机制
pandas的`pd.concat()`默认行为：
- **相同列名** → 自动合并到同一列（如`Unit_ID`、`Incident`、所有分析字段）
- **不同列名** → 自动保留，缺失值填充NaN（如`序号`在VK中为空，`comment_id`在媒体文本中为空）
- **作用**：`required_output_columns`确保所有文件都包含核心分析字段，合并后的数据库有完整列结构

## 通用分析字段配置 (`required_output_columns`)

位置: `config.yaml:187-215`

**通用单元标识列**：
- `Unit_ID` - 议题单元唯一标识
- `Source` - 信源标识（俄总统/俄语媒体/中文媒体/英语媒体/vk/知乎/未知来源）
- `speaker` - 发言人/信源标准化标识
- `Unit_Text` - 议题单元原文
- `Unit_Hash` - 单元文本哈希值（用于去重）
- `processing_status` - 处理状态（SUCCESS/NO_RELEVANT/STAGE_1_FAILED/STAGE_2_FAILED）

**核心事件提取**：
- `Incident` - 核心事件、观点或行动的一句话概括

**六维框架功能分析**（对象格式，包含quote/reason/reasoning_pattern）：
- `Frame_SolutionRecommendation` - 方案建议（优先级1）
- `Frame_ResponsibilityAttribution` - 归因指责（优先级2）
- `Frame_CausalExplanation` - 因果解释（优先级3）
- `Frame_MoralEvaluation` - 道德评价（优先级4）
- `Frame_ProblemDefinition` - 问题建构（优先级5）
- `Frame_ActionStatement` - 事实宣称（优先级6）

**核心分析维度**：
- `Valence` - 情感极性（正面/负面/中立/事实陈述）
- `Evidence_Type` - 证据类型（数据/统计/官方声明/专家观点等）
- `Attribution_Level` - 归因层级（个体/国家/社会/系统/不适用）
- `Temporal_Focus` - 时间焦点（追溯/现状/展望/混合）
- `Primary_Actor_Type` - 主要行动者类型（国家/个人/次国家/国际组织等）
- `Geographic_Scope` - 地理范围（双边/区域/全球/国内/混合）
- `Relationship_Model_Definition` - 关系模式定义（新型关系/传统伙伴/利益关系/未界定）
- `Discourse_Type` - 语段类型（经验性断言/规范性断言/展演性语段）

**作用**：所有处理器通过`_ensure_required_columns()`强制补齐这些列，确保不同批次/语言的输出文件可直接合并为统一数据库。

## 后续建议
- 新增处理器应继承BaseProcessor，复用断点续传和Excel读写逻辑
- 各处理器的prompts位于 `prompts/` 目录，按功能命名
- 知乎处理器可改造为并发架构以提升性能
- 合并多个Excel文件时，pandas会自动对齐相同列名，源特定列（如序号、comment_id）保留但缺失值为NaN
