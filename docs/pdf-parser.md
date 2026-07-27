# PDF 解析层方案

> 本文档定义"解析层"：把原始 PDF 解析为**可溯源的结构化解析结果**，为后续切分策略、证据抽取和 RAG 溯源做准备。
> 解析层只负责"读准、定位准、关系全"，不负责切块、embedding 和模型推理。

## 1. 定位与边界

### 1.1 解析层做什么

- PDF → 逐页渲染 → 版面检测 → OCR / 表格 / 公式识别 → 图片裁剪 → 结构化输出
- 保留每个版面元素的**坐标、页码、阅读顺序、原文内容、置信度**
- 建立**块间关系**：标题层级、图表标题配对、正文对图表的交叉引用

### 1.2 解析层不做什么（留给后续环节）

| 不做 | 归属环节 |
|---|---|
| 按 token / 条款切块 | 切分策略层（后续） |
| 向量化、embedding | 检索层（后续） |
| 图片内容理解、图表语义解读 | MLLM 层（后续） |
| 证据抽取、规则综合 | 准则构建层（后续） |

### 1.3 与 `数据处理.md` 的关系

`数据处理.md` 定义了整体数据分层、block schema 和处理流程总纲。本文档是其中"解析层"的实现细化，输出的 block 结构与 `数据处理.md` 的 schema 对齐：

```
document -> page -> block -> evidence -> rule
```

block 类型统一为：`heading`、`paragraph`、`list`、`table`、`formula`、`figure`、`caption`、`appendix`。

## 2. 技术选型

### 2.1 引擎：PP-StructureV3

| 项 | 选择 | 依据 |
|---|---|---|
| 解析引擎 | PP-StructureV3（Mobile 配置） | 中文 OmniDocBench 流水线派第一；模块化，可按需开关 |
| 版面模型 | PP-DocLayoutV3（产线内置） | 25 类元素 + 实例分割掩码 + 阅读顺序，端到端 |
| OCR | PP-OCRv5（产线内置） | 中英日 + 手写 + 竖排，Paddle 胜任，无需外接 |
| 表格 | SLANeXt（产线内置） | 有线/无线分治，转 HTML |
| 公式 | PP-FormulaNet（产线内置） | 含中文公式、化学方程式，转 LaTeX |
| 图表理解 | **关闭**，交 MLLM | PP-Chart2Table 仅做"图转表"，图片语义理解仍需 MLLM |
| 印章 | 按需关闭 | 毁伤评估场景非核心 |

### 2.2 为何不用 PaddleOCR-VL

- PaddleOCR-VL 的版面阶段用的就是 PP-DocLayoutV3，**版面分析层面与 PP-StructureV3 无差异**。
- PaddleOCR-VL 的优势在 VLM 内容识别（OmniDocBench 96.3%），但该部分 PP-StructureV3 的 OCR/表格/公式已足够；图片理解本就要交 MLLM，VL 替代不了 MLLM。
- PP-StructureV3 用 `paddle_static` 引擎，Windows 原生可跑；VL 的 vLLM/SGLang 路径在 Windows 上需 Docker/WSL。
- 结论：**仅做前置解析时，PP-StructureV3 是更省、更稳、更贴合本机部署的选择。**

### 2.3 硬件与配置（本机）

- GPU：RTX 4070 Ti（16 GB），驱动支持 CUDA 13.0 → 兼容 cu126 运行时
- 推荐配置：**Mobile 套件 + PP-FormulaNet-M + 关闭图表**，显存峰值 ~8.4 GB，V100 实测 ~1.15 s/页，16 GB 卡上安全且快
- 解析与后续 MLLM 分两阶段串行执行，各自独占 GPU，不存在抢显存

## 3. 环境与依赖

环境：conda `dba-py311`（Python 3.11）。

| 包 | 用途 | 安装来源 |
|---|---|---|
| `paddlepaddle-gpu==3.3.0` | 飞桨框架（GPU） | cu126 索引 |
| `paddleocr[doc-parser]>=3.4.0` | PP-StructureV3 产线 API | PyPI |
| `paddlex[ocr]` | PaddleX CLI / 模型下载 | PyPI |
| `pymupdf` | PDF 渲染、文本层检查、原生文本提取 | PyPI |
| `pillow` | 从高分辨率页图生成带冗余边界的视觉块裁图 | PyPI |
| `opencv-python` | 去噪、纠偏、水印弱化 | PyPI |

安装命令：

```bash
conda run -n dba-py311 python -m pip install "paddlepaddle-gpu==3.3.0" \
  -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
conda run -n dba-py311 python -m pip install -U "paddleocr[doc-parser]>=3.4.0" "paddlex[ocr]" pymupdf pillow opencv-python
```

验证：

```bash
conda run -n dba-py311 python -c "import paddle; paddle.utils.run_check()"
conda run -n dba-py311 python -c "from paddleocr import PPStructureV3; print('ok')"
```

> Windows 提示：PP-StructureV3 默认 `paddle_static` 引擎原生可用；`paddleocr[doc-parser]` 若把 CPU 版 `paddlepaddle` 一并装上，需卸载 CPU 版以免覆盖 GPU 版（见第 8 节故障排查）。

## 4. PP-StructureV3 配置

### 4.1 模块开关

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
    res.save_to_markdown(save_path="out/")   # Markdown（含图表标题配对）
    # res.save_to_img(save_path="out/")      # 可视化（调试用）
```

CLI 等价：

```bash
paddleocr pp_structurev3 -i doc.pdf \
  --use_table_recognition True \
  --use_formula_recognition True \
  --use_chart_recognition False \
  --use_seal_recognition False
```

纯英文文档可加 `--text_recognition_model_name en_PP-OCRv4_mobile_rec` 提精度。

### 4.2 标签 → block_type 映射

PP-StructureV3 的 25 类版面标签需归一到 `数据处理.md` 的 block_type：

| PP-StructureV3 label | block_type | 说明 |
|---|---|---|
| `doc_title` | `heading` | 文档标题 |
| `paragraph_title` | `heading` | 段落/章节标题（带编号） |
| `text` | `paragraph` | 正文段落 |
| `list` / `algorithm` | `list` | 列表、算法块 |
| `table` | `table` | 表格区域 |
| `display_formula` / `inline_formula` | `formula` | 公式 |
| `image` / `chart` | `figure` | 图片、图表（图表交 MLLM） |
| `figure_title` / `table_title` / `figure_table_caption` | `caption` | 图表标题 |
| `reference` / `footnote` / `reference_content` | `appendix` 或 `footnote` | 参考文献/脚注 |
| `page_number` / `header` / `footer` | （丢弃或低权） | 页眉页脚页码 |

## 5. 输出结构

### 5.1 PP-StructureV3 原始输出

每个解析结果（每页一个）包含：

| 字段 | 含义 | 解析层用途 |
|---|---|---|
| `label` | 块类型（25 类） | 归一到 block_type |
| `coordinate` | `[xmin, ymin, xmax, ymax]` 像素坐标 | 定位 / 回显 / 裁剪 |
| `polygon_points` | 实例分割掩码点 | 抗倾斜/弯曲贴合 |
| `page_index` | 页码 | 溯源 |
| `order` | 阅读顺序编号 | 后续拼接 |
| `block_content` | 文本 / 表格 HTML / 公式 LaTeX；图片块含图片路径 | 原文回显 / 喂 MLLM |
| `score` | 置信度 | 质量过滤 |

> Markdown 输出里图片已被拼成 `![图3 标题](img_path)`，**图表与标题的配对关系现成可用**，省去自己配对。因此 JSON 与 Markdown 都要保存。

### 5.2 增强后的结构化结果（解析层最终产物）

PP-StructureV3 原始 JSON 直接用**不够**：缺块 ID、缺交叉引用、缺标题编号层级。解析层需做一层后处理（第 6 节），产出增强结构化结果，block schema 在 `数据处理.md` 基础上扩展：

```json
{
  "block_id": "GJB_001_P017_B03",
  "document_id": "GJB_001",
  "page": 17,
  "section_path": ["5", "5.3", "5.3.2"],
  "block_type": "table",
  "bbox": [0.12, 0.18, 0.91, 0.82],
  "bbox_pixel": [88, 142, 668, 640],
  "polygon_points": [[88,142], "..."],
  "order": 11,
  "text": "识别出的内容 / HTML / LaTeX",
  "source_method": "ocr",
  "confidence": 0.86,
  "is_appendix": false,
  "image_crop": "assets/crops/p017_b03_table.png",
  "image_crop_raw": "assets/imgs/img_in_table_box_88_142_668_640.jpg",
  "crop_bbox_pixel": [76, 130, 680, 652],

  "label_norm": "图3 毁伤评估流程图",
  "label_no": "3",
  "caption_of": null,
  "references": []
}
```

新增字段说明（解析层后处理生成）：

| 字段 | 含义 | 生成方式 |
|---|---|---|
| `block_id` | 全局稳定块 ID | `{document_id}_P{page:03d}_B{order:02d}` |
| `bbox` | 归一化坐标 [0,1] | 像素坐标 / 页面宽高 |
| `bbox_pixel` | 原始像素坐标 | 直接来自 PP-StructureV3 |
| `section_path` | 所属标题编号层级 | 见 6.2，沿阅读顺序传递 |
| `label_norm` | 图表标签原文（如"图3 …"） | 见 6.1，从 caption 块抽取 |
| `label_no` | 图表编号（如"3"） | 正则从 label_norm 提取 |
| `caption_of` | caption 指向的 figure/table block_id | 见 6.1 配对 |
| `references` | 本块正文引用的图表 block_id 列表 | 见 6.3 交叉引用 |
| `image_crop_raw` | Paddle 按原始检测框导出的裁图 | 保留原始模型产物用于追溯 |
| `crop_bbox_pixel` | 加冗余后在高分辨率页图上的实际裁剪框 | 原检测框按比例扩张并受 caption 边界约束 |
| `image_crop` | 最终供 MLLM / 回显使用的高分辨率裁图 | 从 `pages/pXXX.png` 按 `crop_bbox_pixel` 重裁 |

> PP-StructureV3 有时会把高置信度 `image` 保留在 `layout_det_res`，但不写入
> `parsing_res_list`。后处理会以 0.90 为默认阈值补回与已有视觉块不重叠的候选，
> 标记 `source_method: "layout"` 并保留检测置信度，避免最终结构遗漏图块。

### 5.3 数据目录结构

```
资料/                           原始 PDF，永不修改
processed/parsed/{doc_id}/
  doc.json                      增强结构化结果（最终产物，喂后续切分）
  structurev3.json              PP-StructureV3 原始 JSON（留底，便于重跑后处理）
  structurev3.md                PP-StructureV3 Markdown（图表配对参考）
  pages/                        逐页渲染图（用于 bbox→可视回溯）
    p017.png
  assets/                       裁剪出的图片/表格/公式子图（喂 MLLM / 回显）
    p017_fig03.png
```

## 6. 解析层后处理（核心补全）

PP-StructureV3 把"图"和"图标题"检测为两个独立块，但**不会把正文里的"如图3所示"链接到图3那个块**。这一层关系必须解析层自己建，且必须在**解析阶段**就建好并存入结构化结果——它是文档结构关系，不是切分策略，且跨页（正文在第3页引用第5页的图很常见）。

### 6.1 图表标题配对（caption ↔ figure/table）

目的：让 `caption` 块知道它属于哪个 `figure`/`table`，反之亦然。

- 取法 A（省事）：直接用 PP-StructureV3 的 Markdown 输出，其中图片已拼成 `![图3 标题](path)`，配对关系现成。
- 取法 B（兜底）：按阅读顺序相邻 + 空间上下贴邻（标题在图正上方/正下方）配对。
- 从 caption 的 `block_content` 抽取标签：正则 `^(图|表|Fig\.?|Table)\s*([0-9A-Za-z\-]+)` → `label_norm`、`label_no`。
- 写入：figure/table 块的 `label_norm`/`label_no`，caption 块的 `caption_of = {figure块block_id}`。

### 6.2 标题编号层级（section_path）

国军标/论文有严格编号体系。引用时用户期望落到"3.2.1 条"而非"第5页某框"。

- 对每个 `paragraph_title` 块，正则抽取编号：`^(\d+(?:\.\d+)*|附录[A-Z]|[A-Z]\.\d+)`。
- 按阅读顺序维护一个标题栈：遇到更深编号压栈，遇到更浅/同级弹栈。
- 后续所有块继承当前栈作为 `section_path`。
- 附录单独处理：记录 `is_appendix`、`appendix_type`（规范性/资料性附录），附录编号纳入 section_path。

> 示例：读到标题"5.3.2 毁伤等级"后，其后的正文块 `section_path = ["5","5.3","5.3.2"]`，直到遇到"5.3.3"或"5.4"。

### 6.3 交叉引用索引（正文 → 图表/附录）

目的：正文文字"如图3所示"能索引到 `figure block_id`，满足"引用到图片表格的文字要有方法索引到它们"。

```
对每个 text/paragraph/heading 块:
  正则匹配文中引用标记:
    图\s?\d+ | 表\s?\d+ | 公式\s?\(\d+\)
    Fig\.?\s*\d+ | Table\s*\d+ | 见第[\d\.]+节 | 见附录[A-Z]
  命中 → 查 6.1 建的 {label_no → block_id} 索引
  写入该块 references: [figure_block_id, table_block_id, ...]
```

- 国军标/论文编号规范，规则法（正则）最可靠；混合格式或不规范的再考虑 LLM 兜底。
- 跨页引用：索引在全文范围建（不止本页），正文在第3页引用第5页的图也能命中。
- 附录引用同理：正文"见附录A"→ `references` 指向附录块，检索时双向展开（检索到正文展开附录，检索到附录返回引用正文）。

### 6.4 坐标归一化

- PP-StructureV3 给像素坐标；`数据处理.md` schema 用归一化 [0,1]。
- 后处理时用页面渲染宽高归一：`bbox = bbox_pixel / [W, H, W, H]`，像素坐标同时保留到 `bbox_pixel`。

### 6.5 原生文本 vs OCR（待试点定夺的增强项）

`数据处理.md` 要求"按页分流"：有文本层用原生文本，扫描页才 OCR。PP-StructureV3 默认对所有页 OCR。

- 基线：统一用 PP-StructureV3 OCR（`source_method: "ocr"`），PP-OCRv5 已足够强。
- 增强（试点若发现文本层论文 OCR 精度不足时再上）：PyMuPDF 检测每页文本层 → 对有文本层页，用原生文本按 bbox 映射回 block，覆盖 OCR 文本，`source_method: "native"`；扫描页保持 OCR。
- 国军标 15/17 为纯图片、395 页无文本层，OCR 不可避免，基线方案对其无影响。

> 该项作为试点的验收项之一（见第 7 节），不阻塞解析层主线。

## 7. 试点与质量验收

不全量处理。先选代表性页（与 `数据处理.md` 第 7 节一致）：扫描国军标正文、密集表格附录、带文本层和水印的标准、双栏公式水印并存的论文、长篇学位论文、英文论文。每类 5~10 页，检查：

- OCR 文字准确度（尤其国军标扫描页）
- 双栏阅读顺序（论文）
- 表格单元格完整度（毁伤等级表）
- 图表标题配对 + 交叉引用命中率（**本解析层重点验收**）
- 标题编号层级 section_path 正确性
- 页码/区域可回溯性（bbox → 原图框选）
- 低质量内容是否被正确标记（`unreadable`，不补写）

试点稳定后再全量解析。

## 8. 数据流总览

```
资料/*.pdf
  │
  ├─[PyMuPDF 逐页渲染] pages/pXXX.png  (+ 文本层检查，供 6.5 增强)
  │
  ├─[PP-StructureV3]
  │     layout(PP-DocLayoutV3) + OCR(PP-OCRv5) + table(SLANeXt) + formula(PP-FormulaNet)
  │     chart=off → figure 区域直接裁剪
  │     → structurev3.json + structurev3.md + assets/*.png
  │
  ├─[解析层后处理]
  │     标签归一(block_type) + block_id 分配 + 坐标归一化
  │     + caption↔figure 配对(6.1) + section_path(6.2)
  │     + 交叉引用索引(6.3) + (可选)原生文本覆盖(6.5)
  │     → doc.json
  │
  └─→ processed/parsed/{doc_id}/doc.json   （后续切分/证据抽取/检索的输入）
        ↓ (后续环节，不在解析层)
   切分策略 → 证据抽取 → RAG 检索 → 带引用溯源的毁伤评估准则
```

## 9. 故障排查

| 现象 | 原因 | 处理 |
|---|---|---|
| `import paddle` 报缺 CUDA / 用了 CPU | `paddleocr` 把 CPU 版 `paddlepaddle` 一并装上覆盖了 GPU 版 | `pip uninstall paddlepaddle` 后重装 `paddlepaddle-gpu==3.3.0 -i cu126` |
| cuDNN 版本警告（compiled 9.9 vs machine 9.5） | paddle wheel 内置 cudnn 9.5，与编译期 9.9 报告不一致 | **非阻塞**：冒烟测试已验证推理正常；若个别算子报错再对齐 cudnn 版本 |
| 模型下载慢/失败 | 默认从 HuggingFace 拉 | `set PADDLE_PDX_MODEL_SOURCE=modelscope` 改用 ModelScope（本机已验证可用） |
| 显存 OOM（16G） | 用了 Server 套件（峰值 17G） | 确认 Mobile 套件；或降 `text_det max_side_limit` 到 1200 |
| Windows 上 PP-StructureV3 报 Conv 维度错 | 旧版 PaddleOCR 缓存模型 | 升级到 `paddleocr>=3.4.0`，删除 `~/.paddlex/official_models/` 重下 |
| 版面检测模型单独加载报错 | 误用旧接口 | 用 `from paddleocr import LayoutDetection`（v3 接口），勿用 `ppstructure/` 旧路径 |
| 同时装了 opencv-python 和 opencv-contrib-python | `paddlex[ocr]` 拉入 contrib | 无害（cv2 仍可用）；如需精简可 `pip uninstall opencv-contrib-python` |

## 10. 环境验证记录（dba-py311）

本机已按第 3 节安装并验证通过（2026-07-27）：

| 项 | 实际版本 | 验证 |
|---|---|---|
| Python | 3.11.15 | conda env |
| paddlepaddle-gpu | 3.3.0 | `paddle.utils.run_check()` 通过，GPU Compute Capability 8.9 |
| paddleocr | 3.7.0 | `from paddleocr import PPStructureV3, LayoutDetection` ok |
| paddlex | 3.7.2 | 模型从 ModelScope 下载成功 |
| pymupdf | 1.28.0 | import fitz ok |
| opencv | 4.10.0 | import cv2 ok |
| 冒烟测试 | LayoutDetection(PP-DocLayoutV3) on `_inspect/page_1.png` | 10 个框，输出含 `coordinate/order/polygon_points` 字段 |

## 11. 批量调用解析层

入口 `python -m src.data_processing`（即 `src/data_processing/__main__.py` -> `pipeline.main()`）。批量由 `parse_pdfs` 编排，要点：

- **整批只加载一次模型**：PPStructureV3 产线构造一次，在所有 PDF 间复用，避免每篇重载版面/OCR/表格/公式模型。
- **断点续跑**：默认 `--skip-existing`，已存在 `doc.json` 的文档直接跳过。中途失败后重跑同命令即续。
- **单篇失败隔离**：某篇出错被记录并跳过，不中断其余文档；退出码非零，便于脚本检测。
- **进度**：每篇打印 `[N/M] doc_id: ok|skipped|failed (页数, 块数, 耗时)`，末尾汇总。

常用命令（conda env `dba-py311`）：

```bash
# 全部论文（跳过已解析的）
conda run -n dba-py311 python -m src.data_processing --papers-only

# 全部资料（国军标 + 论文）
conda run -n dba-py311 python -m src.data_processing --all

# 先冒烟 2 篇验证（强制重跑，不跳过）
conda run -n dba-py311 python -m src.data_processing --papers-only --limit 2 --no-skip-existing

# 指定路径
conda run -n dba-py311 python -m src.data_processing 资料/论文/foo.pdf 资料/论文/bar.pdf

# 只重跑后处理（复用已有 structurev3.json，不加载模型）
conda run -n dba-py311 python -m src.data_processing --all --reuse-detection
```

开关：

| 开关 | 作用 |
|---|---|
| `--papers-only` / `--gjb-only` / `--all` | 枚举 `资料/论文` / `资料/国军标` / 全部 |
| `--skip-existing` / `--no-skip-existing` | 是否跳过已存在 `doc.json`（默认开） |
| `--limit N` | 最多处理 N 篇（冒烟用） |
| `--reuse-detection` | 复用 `structurev3.json`，仅重跑归一/关系/裁图，不加载模型 |
| `--dpi` / `--crop-padding-*` / `--layout-fallback-min-score` | 渲染与裁剪参数（同单篇） |

> 批量解析只加载一次模型，是相对单篇逐次调用最大的性能改进。GPU 单卡串行最稳，不做多进程（会抢显存）。若中途个别篇 OOM，靠失败隔离继续，事后对失败列表重跑即可。

