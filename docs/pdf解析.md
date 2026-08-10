# PDF 解析与 Chunk 流水线

> 本文档定义一条从原始 PDF 到可检索 chunk 的完整流水线：解析层 → 结构化产物（doc.json）→ 后处理关系补全 →（可选）视觉增强 → Chunk 层（章节约束 + 语义切分）。解析层只负责"读准、定位准、关系全"，不负责切块、embedding 和模型推理。

## 1. 流水线总览

### 1.1 整体数据流

本文档覆盖「原始 PDF → 可检索 chunk」的完整流水线。流水线按**数据产品**分阶段，每阶段的实现细节见对应章节：

```text
资料/*.pdf
  │
  ▼
┌──────────────────────────────────────────────────────────────┐
│ ① 解析层 —— 见第 2 章                                       │
│    输入：PDF                                                 │
│    处理：渲染 → 版面检测 → OCR/表格/公式 → 归一清洗 → 关系补全 │
│    产物：doc.json（+ doc.md / 裁图 / relation_validation.jsonl）│
└──────────────────────────────────────────────────────────────┘
  │
  ├─（可选）[② 视觉增强 —— 见第 3 章]
  │       doc.json + crops + 上下文 → visual_enrichment.json
  │       （作为 ③ 的可选输入）
  │
  ▼
┌────────────────────────────────────────────────┐
│ ③ Chunk 层 —— 见第 4 章                        │
│    输入：doc.json（可选读 visual_enrichment.json）│
│    处理：章节约束 + 语义切分                    │
│    产物：chunks.jsonl                          │
└────────────────────────────────────────────────┘
  │
  ▼
┌────────────────────────────────────────────────┐
│ ④ 检索 / 准则构建（后续文档）                  │
│    embedding → 向量索引 → 检索 → 证据抽取 → 规则综合 │
└────────────────────────────────────────────────┘
```

> 本文档详述 ① ② ③；④ 是最终消费方，这里只保留其输入输出契约。

### 1.2 各环节职责与边界

| 环节 | 归属 | 负责 |
|---|---|---|
| 解析层 | 本文档第 2 章 | 读准、定位准、关系全：PDF → doc.json；保留坐标、页码、阅读顺序、置信度；建立标题层级、图表配对、交叉引用、跨页续接 |
| 视觉增强（可选） | 本文档第 3 章 | doc.json + crops + 上下文 → visual_enrichment.json，作 Chunk 层可选输入 |
| Chunk 层 | 本文档第 4 章 | 章节约束 + 语义切分：doc.json → chunks.jsonl；保留 block / page / section 回链 |
| 检索 / 准则构建 | 后续文档 | embedding → 向量索引 → 检索 → 证据抽取 → 规则综合 |

## 2. 解析层

### 2.1 解析层数据流

```text
资料/*.pdf
  │
  ▼
[PyMuPDF 渲染与页元数据]
  │  PDF -> pages/pXXX.png + pages/_render_meta.json
  │  检查文本层、页尺寸、DPI；页图是坐标回溯和裁图依据
  ▼
[PP-StructureV3]
  │  layout(PP-DocLayoutV3) + OCR(PP-OCRv5) + table(SLANeXt) + formula(PP-FormulaNet)
  │  chart=off → figure 区域直接裁剪
  ▼
  structure.json + structure.md + assets/imgs/
  ▼
[归一与清洗]
  │  structure.json -> 统一 Block
  │  标签归一(block_type) + block_id 分配 + 坐标归一化 + 水印/噪声过滤
  ▼
[结构关系构建]
  │  caption <-> figure/table 配对 + section_path 标题层级
  │  + 正文对图表/公式/章节的 references + 跨页 continuation
  ▼
[关系异常检测：非阻塞]
  │  检查目标存在、关系类型、双向图注回链、图注/续接章节兼容性
  │  异常写入 block.quality_flags，并输出 relation_validation.jsonl
  ▼
[视觉裁图与人工审阅]
  │  figure/table + 合法图注 -> assets/crops/ + assets/figures/
  ▼
processed/parsed/{doc_id}/
  ├─ doc.json                 解析层正式机器输入
  ├─ doc.md                   人工审阅视图
  ├─ relation_validation.jsonl 关系异常清单
  └─ assets/crops/            图表回溯与视觉增强输入
```

### 2.2 解析配置

#### 2.2.1 模块开关

```python
from paddleocr import PPStructureV3

pipeline = PPStructureV3()   # 默认 Mobile 套件

output = pipeline.predict(
    "./doc.pdf",
    use_doc_orientation_classify=True,   # 文档方向分类（扫描件有用）
    use_doc_unwarping=False,             # 矫正（按需开，较慢）
    use_textline_orientation=True,       # 文本行方向
    use_table_recognition=True,          # 表格 ✅ 让 Paddle 做
    use_formula_recognition=True,        # 公式 ✅ 让 Paddle 做
    use_chart_recognition=False,         # 图表 ❌ 关闭，交 MLLM
    use_seal_recognition=False,          # 印章（按需）
    use_region_detection=True,           # 版面区域检测（多栏分块，恢复阅读顺序）
)
for res in output:
    res.save_to_json(save_path="out/")       # 结构化 JSON（含坐标）
    res.save_to_markdown(save_path="out/")   # Markdown（含图表标题配对，现成重用）
```

CLI 等价：

```bash
paddleocr pp_structure -i doc.pdf \
  --use_table_recognition True \
  --use_formula_recognition True \
  --use_chart_recognition False \
  --use_seal_recognition False
```

#### 2.2.2 标签 → block_type 映射

PP-StructureV3 的 25 类版面标签归一为 block_type：


| PP-StructureV3 label                                    | block_type              | 说明              |
| ------------------------------------------------------- | ----------------------- | --------------- |
| `doc_title`                                             | `heading`               | 文档标题            |
| `paragraph_title`                                       | `heading`               | 段落/章节标题（带编号）    |
| `text`                                                  | `paragraph`             | 正文段落            |
| `list` / `algorithm`                                    | `list`                  | 列表、算法块          |
| `table`                                                 | `table`                 | 表格区域            |
| `display_formula` / `inline_formula`                    | `formula`               | 公式              |
| `image` / `chart`                                       | `figure`                | 图片、图表（图表交 MLLM） |
| `figure_title` / `table_title` / `figure_table_caption` | `caption`               | 图表标题            |
| `reference` / `footnote` / `reference_content`          | `appendix` 或 `footnote` | 参考文献/脚注         |
| `page_number` / `header` / `footer`                     | （丢弃或低权）                 | 页眉页脚页码          |

### 2.3 解析输出结构

#### 2.3.1 目录结构

每篇 PDF 的输出位于 `processed/parsed/{document_id}/`：

```text
├─ structure.json        PP-StructureV3 原始 JSON（留底，便于重跑后处理）
├─ structure.md          PP-StructureV3 原始 Markdown（图表配对参考）
├─ doc.json              增强结构化结果（最终产物，喂后续切分）
├─ doc.md                最终人工审阅视图
├─ pages/                逐页高分辨率渲染图（坐标回溯与裁图来源）
│  ├─ p001.png
│  └─ _render_meta.json  渲染缓存（DPI + 源 PDF mtime/size + 每页元数据）
└─ assets/
   ├─ imgs/              Paddle 原始模型裁图（留底与排错）
   ├─ crops/             清理后的纯内容裁图（喂 MLLM / 回显）
   └─ figures/           主体 + 全部已配对图注的上下文裁图（人工复核）
```

原则：`structure.*` 是原始产物，`doc.*` 是归一化与关系补全后的正式产物；后续程序**优先读取 `doc.json`**，不应直接依赖 `structure.json` 的内部字段。

#### 2.3.2 四个主文件

| 文件               | 职责                                                                         | 用途                                               |
| ---------------- | -------------------------------------------------------------------------- | ------------------------------------------------ |
| `structure.json` | 模型原始 JSON，按页保存 `parsing_res_list`/`layout_det_res`/`overall_ocr_res` 及模型设置 | 模型排错；调整后处理规则后 `--reuse-detection` 重建；排查漏图/标签/置信度 |
| `doc.json`       | 解析层正式机器输入：标签归一、去噪声、坐标归一化、块 ID、section_path、图文关系、跨页续接                       | 切分、证据抽取、检索的主要输入                                  |
| `structure.md`   | 模型原始 Markdown 视图，内容基本来自 `parsing_res_list`                                 | 观察模型直接输出，诊断产物，不作正式输入                             |
| `doc.md`         | 从 `doc.json` 生成：增强阅读顺序、合并跨页续接、展示补回图片、标注 conditions 与 quality-flags         | 人工检查解析效果的首选                                      |


`doc.json` 相对原始 JSON 完成的增强：

- 去除页眉、页脚和页码等噪声；
- 标签统一为 `heading`/`paragraph`/`table`/`formula`/`figure`/`caption` 等 block_type；
- 坐标转换到 `pages/` 高分辨率页图坐标，并同时保存 `[0,1]` 归一化坐标；
- 为文档、页面和内容块分配可寻址 ID；
- 保存 `section_path`、附录属性和阅读顺序；
- 通过 `caption_ids` / `caption_of` 建立图片与一个或多个图注的关系；
- 通过 `references` 建立正文到图片、表格、公式和章节的引用；
- 通过 `continuation_of` / `continues_to` 连接跨栏、跨页正文；
- 通过 `quality_flags` 标记跨页公式等需要复核的内容；
- 公式编号保存于 `formula_no`，不再把 `(1)`、`(2)` 作为独立公式图片。

#### 2.3.3 页面与图片目录

| 目录                | 内容                                                                    | 要点                                                                                           |
| ----------------- | --------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `pages/`          | 完整逐页渲染图（如 `pages/p003.png`）                                           | 按配置 DPI 渲染；是重裁图像来源；用于 bbox 回显与人工核对版面；不直接作单图输入                                                |
| `assets/imgs/`    | Paddle 自动生成的原始裁图，文件名含模型检测坐标（如 `img_in_image_box_150_318_537_477.jpg`） | 使用模型内部页尺寸与原始检测框，可能裁切过紧、含部分中文图注；用于留底排错，不作默认视觉输入                                               |
| `assets/crops/`   | 从 `pages/` 高分辨率页图重裁的纯内容裁图                                             | 含 figure/table/formula；保留少量安全边界；已配对图注从主体剥离；公式只保留本体（编号在 `formula_no`）；是 MLLM、图片回显和证据定位的默认视觉输入 |
| `assets/figures/` | 图片主体 + 全部已配对图注的上下文裁图（由图片框与关联 caption 坐标并集生成）                          | 双语论文含中英图注；适合人工确认配对与排版上下文；不作默认 MLLM 输入，避免图注文字与 `doc.json` 结构化文本重复                             |


只有成功关联图注的 `figure`/`table` 才生成 `assets/figures/` 对应的上下文裁图。

#### 2.3.4 图片字段对应关系

一个视觉块在 `doc.json` 中可能包含：


| 字段                       | 含义                                |
| ------------------------ | --------------------------------- |
| `image_crop_raw`         | Paddle 原始裁图，对应 `assets/imgs/`     |
| `image_crop`             | 清理后的纯内容裁图，对应 `assets/crops/`      |
| `figure_crop`            | 主体加全部图注的复核裁图，对应 `assets/figures/` |
| `bbox_pixel`             | 模型检测框转换到页图后的坐标                    |
| `crop_bbox_pixel`        | 实际纯内容裁图在页图上的坐标                    |
| `figure_crop_bbox_pixel` | 上下文复核裁图在页图上的坐标                    |


对同一图片，这三个文件承担不同职责，不应互相覆盖。

#### 2.3.5 推荐使用方式

| 使用场景                   | 推荐产物                                           |
| ---------------------- | ---------------------------------------------- |
| 后续切分、证据抽取、RAG          | `doc.json`                                     |
| 人工阅读和解析质量检查            | `doc.md`                                       |
| MLLM 图片、表格、公式输入        | `assets/crops/`                                |
| 人工检查双语图注配对             | `assets/figures/`                              |
| 页面坐标回溯                 | `pages/` + `doc.json`                          |
| 调试 PP-StructureV3 原始输出 | `structure.json`、`structure.md`、`assets/imgs/` |
| 只修改后处理规则后重跑            | `structure.json` + `--reuse-detection`         |

### 2.4 后处理（关系补全）

PP-StructureV3 把"图"和"图标题"检测为两个独立块，却不会把正文里的"如图3所示"链接到图3那个块。这一层关系必须由解析层建立，且必须在**解析阶段**建好存入结构化结果：它是文档结构关系而非切分策略，且存在跨页情况（正文在第3页引用第5页的图很常见）。

#### 2.4.1 图表标题配对（caption ↔ figure/table）

- 取法 A（省事）：直接复用 PP-StructureV3 的 Markdown 输出，图片已拼成 `![图3 标题](path)`，配对关系现成。
- 取法 B（兜底）：按阅读顺序相邻 + 空间上下贴邻（标题在图正上方/正下方）配对。
- 从 caption 的 `block_content` 抽取标签：正则 `^(图|表|Fig\.?|Table)\s*([0-9A-Za-z\-]+)` → `label_norm`、`label_no`。
- 写入：figure/table 块的 `label_norm`/`label_no`，caption 块的 `caption_of = {figure块block_id}`。

#### 2.4.2 标题编号层级（section_path）

国军标/论文有严格编号体系，引用时用户期望落到"3.2.1 条"而非"第5页某框"。

- 对每个 `paragraph_title` 块，正则抽取编号：`^(\d+(?:\.\d+)*|附录[A-Z]|[A-Z]\.\d+)`。
- 按阅读顺序维护标题栈：深入编号压栈，浅/同级弹栈；后续所有块继承当前栈为 `section_path`。
- 附录单独处理：记录 `is_appendix`、`appendix_type`（规范性/资料性），附录编号纳入 section_path。

> 示例：读到标题"5.3.2 毁伤等级"后，其后正文块 `section_path = ["5","5.3","5.3.2"]`，直到遇到"5.3.3"或"5.4"。

#### 2.4.3 交叉引用索引（正文 → 图表/附录）

```
对每个 text/paragraph/heading 块:
  正则匹配文中引用标记:
    图\s?\d+ | 表\s?\d+ | 公式\s?\(\d+\)
    Fig\.?\s*\d+ | Table\s*\d+ | 见第[\d\.]+节 | 见附录[A-Z]
  命中 → 查 2.4.1 建的 {label_no → block_id} 索引
  写入该块 references: [figure_block_id, table_block_id, ...]
```

- 国军标/论文编号规范，规则法（正则）最可靠；混合/不规范格式再考虑 LLM 兜底。
- 跨页引用：索引在全文范围建立，正文在第3页引用第5页的图也能命中。
- 附录引用同理：正文"见附录A" → `references` 指向附录块，检索时双向展开。

#### 2.4.4 坐标归一化

- PP-StructureV3 给像素坐标；数据结构用归一化 `[0,1]`。
- 后处理时用页面渲染宽高归一：`bbox = bbox_pixel / [W, H, W, H]`，像素坐标保留在 `bbox_pixel`。

#### 2.4.5 原生文本 vs OCR（待试点定夺）

`doc.json` 要求"按页分流"：有文本层用原生文本，扫描页才 OCR。PP-StructureV3 默认对所有页 OCR。

- 基线：统一 PP-StructureV3 OCR（`source_method: "ocr"`），PP-OCRv5 已足够强。
- 增强（试点若发现文本层论文 OCR 精度不足再上）：PyMuPDF 检测每页文本层 → 对有文本层页，用原生文本按 bbox 映射回 block，`source_method: "native"`；扫描页保持 OCR。
- 国军标 15/17 为纯图片、395 页无文本层，OCR 不可避免，基线方案无影响。

> 该项是试点验收项之一（见 5.2），不阻塞解析层主线。当前主流程仍以 PP-StructureV3 结果为准，原生文本层覆盖不作为 chunk 前置依赖。

#### 2.4.6 增强字段清单（doc.json 新增）

| 字段                | 含义                                | 生成方式                                      |
| ----------------- | --------------------------------- | ----------------------------------------- |
| `block_id`        | 全局稳定块 ID                          | `{document_id}_P{page:03d}_B{order:02d}`  |
| `bbox`            | 归一化坐标 [0,1]                       | 像素坐标 / 页面宽高                               |
| `bbox_pixel`      | 页图上的原始像素坐标                        | PP-StructureV3 检测框转换到页图                   |
| `section_path`    | 所属标题编号层级                          | 沿阅读顺序由标题栈传递（2.4.2）                          |
| `label_norm`      | 图表标签原文（如"图3 …"）                   | 从 caption 块正则抽取（2.4.1）                      |
| `label_no`        | 图表编号（如"3"）                        | 正则从 label_norm 提取                         |
| `caption_of`      | caption 指向的 figure/table block_id | 2.4.1 配对                                    |
| `references`      | 本块引用的图表 block_id 列表               | 2.4.3 交叉引用                                  |
| `source_method`   | `ocr` / `layout` / `native`       | 见 2.4.5 及下方补图规则                             |
| `image_crop_raw`  | Paddle 原始导出裁图                     | 保留原始模型产物供追溯                               |
| `crop_bbox_pixel` | 加冗余后在高分辨率页图上的实际裁剪框                | 原检测框按比例扩张并受 caption 边界约束                  |
| `image_crop`      | 最终供 MLLM / 回显的高分辨率裁图              | 从 `pages/pXXX.png` 按 `crop_bbox_pixel` 重裁 |


> PP-StructureV3 有时把高置信度 `image` 保留在 `layout_det_res` 但不写入 `parsing_res_list`。后处理以 0.90 为默认阈值补回与已有视觉块不重叠的候选，标记 `source_method: "layout"` 并保留检测置信度，避免结构遗漏图块。

## 3. 视觉增强（MLLM 图片解析）

### 3.1 视觉增强数据流

```text
doc.json + assets/crops/
  │
  ▼
[上下文组装]
  │  所属章节 + 图注 + 引用正文 + 相邻正文 + 已有块文本
  ▼
[LM Studio 视觉模型]（OpenAI 兼容 API）
  │  对每个 figure/table 块生成描述
  ▼
visual_enrichment.json
  │  items[] = { block_id, status, description, error }
  ▼
[Chunk 层可选读取] -> visual_text / visual_assets（见第 4 章）
```

### 3.2 定位与边界

视觉增强是**可选环节**：读取解析层正式产物 `doc.json` 和 `assets/crops/`，对 `figure` / `table` 块调用本地视觉模型（LM Studio，OpenAI 兼容 API）生成图片描述，写入独立的 `visual_enrichment.json`。

- 它**不修改解析事实**：`doc.json` 仍是唯一事实来源，`visual_enrichment.json` 是可替换的检索辅助产物。
- 与解析层的边界：解析层只负责"读准、定位准、关系全"（见 1.2），**不做图表语义解读**；图片/表格内容理解由本章（MLLM 层）负责。
- 与 Chunk 层的关系：Chunk 层可选读取 `visual_enrichment.json`，把描述写入 chunk 的 `visual_text`（低权重检索辅助文本），不混入解析事实 `text`。

### 3.3 输入与产物

| 方向 | 内容 | 来源/去向 |
|---|---|---|
| 输入 | `doc.json` | 提供 `block_id`、`block_type`、`page`、`section_path`、`caption_ids`、`references`、`image_crop` |
| 输入 | `assets/crops/` | 清理后的高分辨率纯内容裁图，直接喂给视觉模型 |
| 输入 | 上下文 | 所属章节、图注、引用正文、同章节相邻正文、已有块文本（用于提示词） |
| 产物 | `visual_enrichment.json` | 与 `doc.json` 同目录，可选供 Chunk 层读取 |

`visual_enrichment.json` 结构：

| 字段 | 含义 |
|---|---|
| `schema_version` | 产物契约版本，当前为 `1` |
| `document_id` | 来源文档 ID |
| `source_doc` | 来源 `doc.json` 路径 |
| `items[]` | 每个 `figure` / `table` 块一条记录 |
| `items[].block_id` | 对应解析块 |
| `items[].status` | `ok` / `error` |
| `items[].description` | `ok` 时的图片描述（1~3 句，最多 150 字） |
| `items[].error` | `error` 时的失败原因 |

### 3.4 执行

入口 `src/data_processing/visual_enrichment.py`，基于 LM Studio 本地 OpenAI 兼容 API（默认 `http://localhost:1234/v1`），需用 `--model` 指定已加载的视觉模型。

```powershell
# 处理单篇文档
conda run -n dba-py311 python -m src.data_processing.visual_enrichment ^
  processed/parsed/2-电子系统/doc.json --model <vision-model>

# 处理全部解析文档
conda run -n dba-py311 python -m src.data_processing.visual_enrichment --all --model <vision-model>

# 强制重跑已有成功结果
conda run -n dba-py311 python -m src.data_processing.visual_enrichment --all --model <vision-model> --force
```

| 开关 | 作用 |
|---|---|
| `doc.json` 路径列表 / `--all` | 处理指定文档或 `processed/parsed` 下全部文档 |
| `--base-url` | LM Studio OpenAI 兼容 API 地址，默认 `http://localhost:1234/v1` |
| `--model` | 已加载的视觉模型标识（必填） |
| `--timeout` / `--max-tokens` | 单次调用超时与输出上限 |
| `--force` | 重新处理已有成功结果 |
| `--limit` | 最多处理 N 篇（冒烟用） |

要点：

- **断点续跑**：默认跳过已有 `status=ok` 的块，重跑同命令即续。
- **单块失败隔离**：一张图失败不中断整篇文档，记录 `error`，其余块正常处理。
- 输出会做 **150 字截断**，尽量按 `。！？.!?` 句边界截断，避免留下断句。

### 3.5 提示词与输出约束

- 系统提示只要求"提取可确认的视觉内容"，不猜测、不补常识、不输出 JSON/Markdown 标题。
- 描述聚焦关键对象及可见关系；图注只作辅助，不单纯复述图注。
- 兼容旧 JSON 输出：接受代码块围栏或 `{"description": ...}` 老格式，便于中断任务续跑。

### 3.6 与 Chunk 层的衔接

- Chunk 层读取 `doc.json` 时可选读取 `visual_enrichment.json`，将 `description` 写入 `visual_text`。
- 视觉增强缺失、失败或裁图缺失时，Chunk 层继续生成文本 chunk，并标记 `visual_unavailable`（见 4.3）。
- 后续增强（P1）：为视觉增强记录模型名、提示词版本和生成时间（见 4.5）。

## 4. Chunk 层

Chunk 层读取解析层正式产物 `doc.json`，可选读取同目录的 `visual_enrichment.json`，独立写出 `chunks.jsonl`，不改写解析事实。

### 4.1 Chunk 数据流

```text
doc.json
  │
  ├─（可选）visual_enrichment.json
  ▼
[构建单元]
  │  以段落/列表/公式/表格/图/图注为语义单元
  │  跨页续接块、图表与图注、公式与解释正文保持强绑定
  ▼
[章节约束 + 打包]
  │  section_path 硬边界 + 目标 600 / 上限 800 字 + 80 字重叠
  ▼
[最低校验（P0）]
  │  回链完整、坐标存在、视觉不可用标记
  ▼
chunks.jsonl
```

### 4.2 策略

- 使用完整 `section_path` 作为硬边界，不跨章节合并。
- 仅包含标题的父级节点不生成独立 chunk；其完整标题路径写入每个有正文的后代 chunk。
- `embedding_text` 是向量化输入，包含文档名、完整标题路径及正文；`text` 只保留正文，不含 Markdown 标题标记。
- 以段落、列表、公式、表格、图和图注为语义单元。
- 标题与本章节首个内容单元、图表与图注、跨页续接块、正文引用同章节公式时公式与解释性正文保持**强绑定**。
- 默认目标 600 字、上限 800 字；单个表格、公式及强绑定单元允许超限。
- 只有超长普通文本会按句切分；相邻 chunk 默认保留 80 字轻量重叠。
- MLLM 描述写入 `visual_text`，不混入解析事实 `text`；视觉增强缺失、失败或裁图缺失时继续生成文本 chunk，并标记 `visual_unavailable`。

### 4.3 chunk 前最低校验（P0）

chunk 程序启动时检查并记录以下问题，但不因单个异常块中止整篇文档：

1. `block_id` 在文档内唯一，且 `document_id`、页码和坐标存在。
2. `references`、`caption_of`、`caption_ids`、`continuation_of`、`continues_to` 指向存在的块；无效关系进入质量告警。
3. 图表块缺裁图、视觉增强缺失或 MLLM 失败时，保留图表正文/OCR/caption，标记 `visual_unavailable`。
4. `section_path=[]` 允许存在（封面、前置页或未归属内容），不得简单丢弃。
5. 输出 chunk 必须保留 `document_id`、`block_ids`、页码、`section_path`、来源文件和图表回链信息。

### 4.4 输出契约

每行是一个 JSON 对象，核心字段：


| 字段                        | 含义                         |
| ------------------------- | -------------------------- |
| `schema_version`          | Chunk 字段契约版本，当前为 `2`       |
| `chunk_id`                | 文档内稳定顺序 ID                 |
| `text`                    | 来自 `doc.json` 的主证据文本       |
| `embedding_text`          | 用于向量化的规范化文本，含文档名、完整标题路径和正文 |
| `visual_text`             | 可选、低权重的 MLLM 检索辅助文本        |
| `overlap_text`            | 上一个 chunk 的轻量文本重叠          |
| `block_ids`               | 主内容对应的解析块                  |
| `overlap_block_ids`       | 重叠文本对应的解析块                 |
| `page_start` / `page_end` | 来源页范围                      |
| `section_path`            | 所属章节路径                     |
| `heading_path`            | 人类可读的完整标题路径，供检索上下文和展示使用    |
| `references`              | 正文引用的块 ID                  |
| `visual_assets`           | chunk 内或被引用图表的裁图、描述和关系     |
| `quality_flags`           | 无效关系、视觉不可用等非阻塞告警           |

### 4.5 实现优先级

- **P0：** 冻结字段契约；实现章节约束的最小 chunk；保留 block/page/source 回链；允许视觉增强缺失。
- **P1：** 关系完整性校验、表格/公式专门处理、chunk 统计和小规模召回验证；为视觉增强记录模型名、提示词版本和生成时间。
- **P2：** 语义模型辅助边界、行列级表格切分、增量重建、跨文档去重和系统化评估。

### 4.6 使用

```powershell
# 处理指定文档
conda run -n dba-py311 python scripts/build_chunks.py processed/parsed/2-电子系统/doc.json

# 处理全部解析文档
conda run -n dba-py311 python scripts/build_chunks.py --all
```

可用 `--target-chars`、`--max-chars` 和 `--overlap-chars` 调整首版字符阈值。

## 5. 运行与验收

### 5.1 批量调用解析层

入口 `python -m src.data_processing`（`src/data_processing/__main__.py` → `pipeline.main()`），批量由 `parse_pdfs` 编排，要点：

- **整批只加载一次模型**：PPStructureV3 产线构造一次，在所有 PDF 间复用。
- **断点续跑**：默认 `--skip-existing`，已存在 `doc.json` 的文档直接跳过，重跑同命令即续。
- **单篇失败隔离**：某篇出错被记录并跳过，不中断其余文档；退出码非零便于脚本检测。
- **进度**：每篇打印 `[N/M] doc_id: ok|skipped|failed (页数, 块数, 耗时)`，末尾汇总。
- **渲染结果缓存**：`render_pdf` 落盘 `pages/_render_meta.json`；命中（DPI 一致、源 PDF 未改、页图齐全）则复用页图，`--reuse-detection` 重跑后处理不重复转图。强制重渲染可删 `pages/` 或换 `--dpi`。

```bash
# 全部论文（跳过已解析的）
conda run -n dba-py311 python -m src.data_processing --papers-only

# 全部资料（国军标 + 论文）
conda run -n dba-py311 python -m src.data_processing --all

# 先冒烟 2 篇验证（强制重跑，不跳过）
conda run -n dba-py311 python -m src.data_processing --papers-only --limit 2 --no-skip-existing

# 指定路径
conda run -n dba-py311 python -m src.data_processing 资料/论文/foo.pdf 资料/论文/bar.pdf

# 只重跑后处理（复用已有 structure.json，不加载模型）
conda run -n dba-py311 python -m src.data_processing --all --reuse-detection
```


| 开关                                                           | 作用                                    |
| ------------------------------------------------------------ | ------------------------------------- |
| `--papers-only` / `--gjb-only` / `--all`                     | 枚举 `资料/论文` / `资料/国军标` / 全部            |
| `--skip-existing` / `--no-skip-existing`                     | 是否跳过已存在 `doc.json`（默认开）               |
| `--limit N`                                                  | 最多处理 N 篇（冒烟用）                         |
| `--reuse-detection`                                          | 复用 `structure.json`，仅重跑归一/关系/裁图，不加载模型 |
| `--dpi` / `--crop-padding-*` / `--layout-fallback-min-score` | 渲染与裁剪参数（同单篇）                          |


GPU 单卡串行最稳，不做多进程（会抢显存）；个别篇 OOM 靠失败隔离继续，事后重跑即可。

### 5.2 试点与质量验收

不全量处理，先选代表性页（每类 5~10 页）：扫描国军标正文、密集表格附录、带文本层和水印的标准、双栏公式水印并存的论文、长篇学位论文、英文论文。检查：

- OCR 文字准确度（尤其国军标扫描页）
- 双栏阅读顺序（论文）
- 表格单元格完整度（毁伤等级表）
- 图表标题配对 + 交叉引用命中率（**解析层重点验收**）
- 标题编号层级 section_path 正确性
- 页码/区域可回溯性（bbox → 原图框选）
- 低质量内容是否被正确标记（`unreadable`，不补写）

试点稳定后再全量解析。

### 5.3 产物异常定位

| 看到的问题                | 首先检查                                                             | 处理位置             |
| -------------------- | ---------------------------------------------------------------- | ---------------- |
| OCR 文字、表格或公式错误       | `structure.json`、页图                                              | 检测模型或归一化         |
| 标题层级或章节归属错误          | `doc.json` 的 `section_path`                                      | 后处理 2.4.2 标题栈      |
| 图表和图注错配              | `relation_validation.jsonl`、双方 `caption_of` / `caption_ids`、页图坐标 | 后处理 2.4.1 配对       |
| 正文引用未带出公式            | `doc.json.references` 与 `chunks.jsonl.block_ids`                 | chunk 强绑定规则（4.2） |
| chunk 跨章节、空正文或缺标题上下文 | `chunks.jsonl`                                                   | chunk 构建规则       |
| 图表描述缺失或裁图不可用         | `visual_enrichment.json`、`assets/crops`                          | 视觉增强或裁图          |


异常检测不替代人工审阅。它把全量人工筛查缩小为两类样本：`relation_validation.jsonl` 中的告警记录，以及少量无告警的随机抽样。每条告警含源/目标 block ID 与页码，可直接回到 `doc.json` 和页图确认。

## 6. 故障排查与运行环境（附录）

### 6.1 故障排查

| 现象                                         | 原因                                            | 处理                                                                     |
| ------------------------------------------ | --------------------------------------------- | ---------------------------------------------------------------------- |
| `import paddle` 报缺 CUDA / 用了 CPU           | `paddleocr` 装上了 CPU 版 `paddlepaddle` 覆盖 GPU 版 | `pip uninstall paddlepaddle` 后重装 `paddlepaddle-gpu==3.3.0 -i cu126`    |
| cuDNN 版本警告（compiled 9.9 vs machine 9.5）    | paddle wheel 内置 cudnn 9.5，与编译期报告不一致           | **非阻塞**：冒烟测试已验证推理正常；个别算子报错再对齐 cudnn 版本                                 |
| 模型下载慢/失败                                   | 默认从 HuggingFace 拉取                            | `set PADDLE_PDX_MODEL_SOURCE=modelscope` 改用 ModelScope（已验证可用）          |
| 显存 OOM（16G）                                | 用了 Server 套件（峰值 17G）                          | 确认 Mobile 套件；或降 `text_det max_side_limit` 到 1200                       |
| Windows 上 PP-StructureV3 报 Conv 维度错        | 旧版 PaddleOCR 缓存模型                             | 升级 `paddleocr>=3.4.0`，删除 `~/.paddlex/official_models/` 重下              |
| 版面检测模型单独加载报错                               | 误用旧接口                                         | 用 `from paddleocr import LayoutDetection`（v3 接口），勿用 `ppstructure/` 旧路径 |
| 同时装了 opencv-python 和 opencv-contrib-python | `paddlex[ocr]` 拉入 contrib                     | 无害；如需精简 `pip uninstall opencv-contrib-python`                          |

### 6.2 本机环境验证记录（dba-py311）

| 项                | 实际版本                                                     | 验证                                                        |
| ---------------- | -------------------------------------------------------- | --------------------------------------------------------- |
| Python           | 3.11.15                                                  | conda env                                                 |
| paddlepaddle-gpu | 3.3.0                                                    | `paddle.utils.run_check()` 通过，GPU Compute Capability 8.9  |
| paddleocr        | 3.7.0                                                    | `from paddleocr import PPStructureV3, LayoutDetection` ok |
| paddlex          | 3.7.2                                                    | 模型从 ModelScope 下载成功                                       |
| pymupdf          | 1.28.0                                                   | import fitz ok                                            |
| opencv           | 4.10.0                                                   | import cv2 ok                                             |
| 冒烟测试             | LayoutDetection(PP-DocLayoutV3) on `_inspect/page_1.png` | 10 个框，输出含 `coordinate/order/polygon_points` 字段            |

