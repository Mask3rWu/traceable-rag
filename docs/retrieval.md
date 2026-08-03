# 混合检索说明

当前检索层实现 Dense、中文 BM25 与 RRF 融合。Reranker 只保留接口，当前使用
`NoopReranker`，不会改变融合排序。

## 数据边界

- `SearchResult` 只保存 `chunk_id`、`content_hash`、通道排名和分数，不复制正文与溯源字段。
- 需要展示正文时通过 `ChunkCatalog` 从 `chunks.jsonl` 一次性加载。
- Dense、BM25 和本地 chunk 的 `content_hash` 必须一致；不一致时检索或评测直接失败。
- BM25 使用 `source_file + document_id + heading_path + text`，不索引模型生成的 `visual_text`。

## 构建索引

在项目 Python 环境安装检索依赖：

```powershell
conda run -n dba-py311 python -m pip install -r requirements-retrieval.txt
```

Dense 索引沿用现有构建命令，BM25 索引单独生成：

```powershell
conda run -n dba-py311 python scripts/build_embeddings.py --all
conda run -n dba-py311 python scripts/build_bm25.py --all
```

BM25 索引默认写入 `processed/retrieval/bm25/`，属于可重建产物。

## 对比评测

先抽样验证完整链路：

```powershell
conda run -n dba-py311 python scripts/evaluate_retrieval.py --max-questions 100
```

再运行完整评测并保存报告：

```powershell
conda run -n dba-py311 python scripts/evaluate_retrieval.py `
  --output processed/retrieval/eval-baseline.json
```

指标包括 `evidence_recall`、`hit_rate`、`complete_recall` 和 `MRR`。评测只计算
直接召回的 `block_ids`，不包含相邻块或引用关系扩展。

### 当前基线

在 3779 个 chunk、`eval/v1` 全部 2083 题上，候选数为 50、RRF 等权且
`rank_constant=60` 时：

| 方法 | Evidence Recall@10 | Evidence Recall@50 | MRR@10 |
|---|---:|---:|---:|
| Dense | 0.6942 | 0.8320 | 0.4998 |
| BM25 | 0.9476 | 0.9851 | 0.8222 |
| RRF | 0.8862 | 0.9789 | 0.6631 |

该结果不能直接用于选择生产权重。当前 2058 道批量题由证据原文抽取主题后套用
固定模板，字面泄漏明显，主要评估原文片段回查。单独编写的 `2-电子系统` 25 题
上，RRF 的 MRR 为 0.8300，高于 BM25 的 0.8158 和 Dense 的 0.7867，但样本量
不足。当前应保留等权 RRF 作为中性基线，待补充独立改写、跨文档和困难负例后再
调整权重。

## 人工检查

```powershell
conda run -n dba-py311 python scripts/search_retrieval.py `
  "坦克履带断裂对运动能力有什么影响" --method rrf --limit 10
```

`--method` 可取 `dense`、`bm25` 或 `rrf`。
