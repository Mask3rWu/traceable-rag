# 批量 PDF 解析 runner 实现方案

## 目标
让 30 篇论文（+ 国军标）只加载一次 PP-StructureV3 模型，批量解析；跳过已解析、单篇失败隔离、打印进度，并能在出错后断点续跑。

## 问题根因
`detect.py:_build_pipeline()` 在每次 `detect_pdf()` 调用里都重新 `PPStructureV3(...)` 构造，触发版面/OCR/表格/公式模型加载。30 篇 = 30 次重载。`pipeline.main()` 循环无 try/except、无跳过逻辑。

## 改动清单

### 1. `src/data_processing/detect.py` — 支持注入 pipeline
- 把 `_build_pipeline` 暴露为公开 `build_pipeline(config)`（保留私有别名或直接重命名）。
- `detect_pdf(pdf_path, out_dir, config=None, *, pipeline=None)`：
  - `pipeline is None` → 自行 `build_pipeline`（保持单篇旧行为）。
  - 否则复用传入的 pipeline。其余逻辑不动。

### 2. `src/data_processing/pipeline.py` — 串接 + 批量函数 + CLI
- `parse_pdf(...)` 新增 `pipeline=None` 参数，透传给 `detect_pdf`（仅 `not reuse_detection` 时用到）。
- 新增 `parse_pdfs(pdf_paths, *, config=None, skip_existing=True, limit=None, reuse_detection=False) -> dict`：
  - 非复用模式：`build_pipeline(config)` **只建一次**，全批复用。
  - 逐篇：
    - `doc_id = doc_id_from_path(p)`；`skip_existing` 且 `doc.json` 已存在 → 记 skipped，跳过。
    - `try: parse_pdf(p, config, pipeline=pipe, reuse_detection=...)`；`except Exception as e:` 记 failed + 错误信息，**继续**下一篇。
    - 进度行：`[N/M] doc_id ... ok|skipped|failed (页数/块数, 耗时)`。
    - `limit` 截断数量（便于冒烟）。
  - 返回 `{total, ok, skipped, failed:[{doc_id, source, error}]}`。
- `build_parser()`：
  - `pdf` 改 `nargs="*`（零或多个显式路径）。
  - 新增 `--all` / `--papers-only` / `--gjb-only`（用 `list_input_pdfs` / `PAPER_DIR` / `GJB_DIR` 枚举）。
  - 新增 `--skip-existing`（默认 True）/ `--no-skip-existing`。
  - 新增 `--limit N`。
  - 保留 `--reuse-detection` 及现有渲染/裁剪参数。
  - 校验：无路径且无枚举开关 → 报错退出。
- `main()`：解析路径列表 → `parse_pdfs(...)` → 打印汇总；任一 failed 返回非零退出码（便于脚本检测）。
  - `--reuse-detection` 时不建 pipeline（无需模型）。

### 3. `src/paths.py` — 小补
- `list_input_pdfs()` 已存在，直接复用。可加 `list_paper_pdfs()` / `list_gjb_pdfs()` 两个小函数供枚举开关用（或内联 glob，二选一；倾向加函数保持单一数据源）。

### 4. 测试 `tests/test_pipeline.py`
- 新增 `test_parse_pdfs_skips_existing_and_isolates_failure`：
  - 用两个临时 PDF + 预置 `structurev3.json`（走 `reuse_detection=True`，不依赖 paddle）。
  - 断言：两篇都产 `doc.json`；第二篇跑前已有 `doc.json` 时被 skip；人为让其中一篇 `parse_pdf` 抛错（monkeypatch）时另一篇仍成功；汇总字段正确。
- 单篇 `parse_pdf` 旧行为不变 → 现有测试不动。

### 5. 文档 `docs/pdf-parser.md`
- 在 CLI 示例附近加"批量调用"小节：`--all` / `--papers-only` / `--skip-existing` / `--limit` 用法，以及断点续跑说明（失败后重跑同命令即续）。

## 不做
- 不改 schema/normalize/relations/crop/render 逻辑。
- 不引入多进程/异步（GPU 单卡串行最稳，多进程会抢显存）。
- 不做断点文件持久化（靠 `--skip-existing` 天然续跑即可）。

## 验证步骤
1. `conda run -n dba-py311 python -m pytest tests/ -q` 全绿（含新批量测试）。
2. 冒烟：`python -m src.data_processing --papers-only --limit 2 --no-skip-existing` 跑 2 篇，确认只加载一次模型、进度行正确。
3. 全量：`python -m src.data_processing --papers-only --skip-existing`（后台跑，监控进度与失败列表），跑完查 `processed/parsed/` 下 `doc.json` 数量。

## 风险
- PPStructureV3 跨 30 篇复用可能 GPU 显存碎片化 → 若中途 OOM，靠失败隔离继续，事后对 failed 列表重跑（`--skip-existing` 跳过已成的）。
- 单篇耗时随页数差异大，长篇学位论文会更久；属正常。
