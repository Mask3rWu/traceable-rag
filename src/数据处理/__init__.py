"""数据处理模块：PDF 解析层。

对应 pdf-parser.md：
- render: PDF->逐页图 + 文本层检查
- detect: PP-StructureV3 包装
- normalize: 原始输出->归一 block
- relations: 块间关系（caption/section_path/交叉引用）
- pipeline: 端到端编排
"""
