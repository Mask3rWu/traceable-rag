"""配置：模型名、PP-StructureV3 模块开关、阈值、环境变量。

对应 pdf-parser.md §4。Mobile 套件 + 关闭图表/印章，适配 16G 显卡。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class ParseConfig:
    """PP-StructureV3 调用配置。"""

    # 显式模型名，避免 PaddleOCR 升级后默认切到 Server/Plus-L 套件。
    layout_detection_model_name: str = "PP-DocLayoutV3"
    text_detection_model_name: str = "PP-OCRv5_mobile_det"
    text_recognition_model_name: str = "PP-OCRv5_mobile_rec"
    # PaddleOCR 3.7 中对应方案文档所述 PP-FormulaNet-M 的当前模型名。
    formula_recognition_model_name: str = "PP-FormulaNet_plus-M"

    # 模块开关（对应 pdf-parser.md §4.1）
    use_doc_orientation_classify: bool = True
    use_doc_unwarping: bool = False
    use_textline_orientation: bool = True
    use_table_recognition: bool = True
    use_formula_recognition: bool = True
    use_chart_recognition: bool = False  # 图表交 MLLM
    use_seal_recognition: bool = False
    use_region_detection: bool = True  # 多栏分块，恢复阅读顺序

    # 模型源：本机已验证 ModelScope 可用
    model_source: str = "modelscope"

    # 页面渲染 DPI（用于 OCR 与回溯）
    render_dpi: int = 200

    # 最终视觉块从 page_image 重裁时的冗余。ratio 相对原检测框宽/高。
    crop_padding_x_ratio: float = 0.02
    crop_padding_top_ratio: float = 0.02
    crop_padding_bottom_ratio: float = 0.08
    crop_padding_min_px: int = 12
    crop_caption_gap_px: int = 1

    # parsing_res_list 偶尔漏掉 layout_det_res 中已检出的图片候选。
    layout_visual_fallback_min_score: float = 0.90

    def to_pipeline_kwargs(self) -> dict:
        """转为 PPStructureV3() 的显式模型参数。"""
        return {
            "layout_detection_model_name": self.layout_detection_model_name,
            "text_detection_model_name": self.text_detection_model_name,
            "text_recognition_model_name": self.text_recognition_model_name,
            "formula_recognition_model_name": self.formula_recognition_model_name,
        }

    def to_predict_kwargs(self) -> dict:
        """转为 PPStructureV3.predict() 的 kwargs。"""
        return {
            "use_doc_orientation_classify": self.use_doc_orientation_classify,
            "use_doc_unwarping": self.use_doc_unwarping,
            "use_textline_orientation": self.use_textline_orientation,
            "use_table_recognition": self.use_table_recognition,
            "use_formula_recognition": self.use_formula_recognition,
            "use_chart_recognition": self.use_chart_recognition,
            "use_seal_recognition": self.use_seal_recognition,
            "use_region_detection": self.use_region_detection,
        }


def apply_env(config: ParseConfig) -> None:
    """设置模型源等环境变量（必须在 import paddleocr 前生效）。"""
    os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", config.model_source)
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")


DEFAULT = ParseConfig()
