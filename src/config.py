"""配置：模型名、PP-StructureV3 模块开关、阈值、环境变量。

对应 pdf-parser.md §4。Mobile 套件 + 关闭图表/印章，适配 16G 显卡。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from src.paths import PROJECT_ROOT


class ConfigError(ValueError):
    """Raised when required application configuration is missing or invalid."""


def _positive_int_env(name: str, *, default: int) -> int:
    raw_value = os.getenv(name, str(default)).strip()
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ConfigError(f"{name} must be greater than zero")
    return value


def _bool_env(name: str, *, default: bool = False) -> bool:
    raw_value = os.getenv(name, str(default)).strip().lower()
    if raw_value in {"1", "true", "yes", "on"}:
        return True
    if raw_value in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{name} must be a boolean")


@dataclass(frozen=True)
class EmbeddingConfig:
    """Connection settings for an OpenAI-compatible embedding model."""

    model: str
    base_url: str
    api_key: str = field(repr=False)
    dimension: int = 1024
    batch_size: int = 32

    @classmethod
    def from_env(cls, env_file: str | Path | None = None) -> "EmbeddingConfig":
        """Load embedding settings, with process variables overriding ``.env``."""
        dotenv_path = Path(env_file) if env_file is not None else PROJECT_ROOT / ".env"
        load_dotenv(dotenv_path=dotenv_path, override=False)

        names = {
            "model": "EMBEDDING_MODEL",
            "base_url": "EMBEDDING_BASE_URL",
            "api_key": "EMBEDDING_API_KEY",
        }
        values = {
            field_name: os.getenv(env_name, "").strip()
            for field_name, env_name in names.items()
        }
        missing = [names[field_name] for field_name, value in values.items() if not value]
        if missing:
            raise ConfigError(
                "Missing required embedding configuration: " + ", ".join(missing)
            )

        values["base_url"] = values["base_url"].rstrip("/")
        dimension = _positive_int_env("EMBEDDING_DIMENSION", default=1024)
        batch_size = _positive_int_env("EMBEDDING_BATCH_SIZE", default=32)
        return cls(**values, dimension=dimension, batch_size=batch_size)


@dataclass(frozen=True)
class DatabaseConfig:
    """Connection settings for the project's PostgreSQL/pgvector database."""

    host: str
    port: int
    database: str
    user: str
    password: str = field(repr=False)

    @classmethod
    def from_env(cls, env_file: str | Path | None = None) -> "DatabaseConfig":
        """Load database settings from the project ``.env`` file."""
        dotenv_path = Path(env_file) if env_file is not None else PROJECT_ROOT / ".env"
        load_dotenv(dotenv_path=dotenv_path, override=False)

        names = {
            "host": "DB_HOST",
            "port": "DB_PORT",
            "database": "DB_NAME",
            "user": "DB_USER",
            "password": "DB_PASSWORD",
        }
        values = {
            field_name: os.getenv(env_name, "").strip()
            for field_name, env_name in names.items()
        }
        missing = [names[field_name] for field_name, value in values.items() if not value]
        if missing:
            raise ConfigError(
                "Missing required database configuration: " + ", ".join(missing)
            )

        try:
            port = int(values.pop("port"))
        except ValueError as exc:
            raise ConfigError("DB_PORT must be an integer") from exc
        if not 1 <= port <= 65535:
            raise ConfigError("DB_PORT must be between 1 and 65535")
        return cls(port=port, **values)

    def connection_kwargs(self) -> dict[str, str | int]:
        """Return keyword arguments accepted by ``psycopg.connect``."""
        return {
            "host": self.host,
            "port": self.port,
            "dbname": self.database,
            "user": self.user,
            "password": self.password,
        }


@dataclass(frozen=True)
class ResearchModelConfig:
    """Connection settings for an OpenAI-compatible research model."""

    model: str
    base_url: str
    api_key: str = field(repr=False)
    max_queries: int = 4
    evidence_limit: int = 10
    max_steps: int = 12
    fast_max_steps: int = 8
    worker_max_steps: int = 30
    supervisor_max_steps: int = 12
    retrieval_top_k: int = 8
    max_evidence_reads: int = 40
    max_workers: int = 4
    max_subtasks: int = 8
    document_max_chars: int = 6000
    chapter_max_chars: int = 1600
    chapter_max_claims: int = 10
    chapter_max_decisions: int = 4
    langfuse_enabled: bool = False
    langfuse_public_key: str = field(default="", repr=False)
    langfuse_secret_key: str = field(default="", repr=False)
    langfuse_base_url: str = "https://cloud.langfuse.com"

    @classmethod
    def from_env(cls, env_file: str | Path | None = None) -> "ResearchModelConfig":
        dotenv_path = Path(env_file) if env_file is not None else PROJECT_ROOT / ".env"
        load_dotenv(dotenv_path=dotenv_path, override=False)

        names = {
            "model": "RESEARCH_MODEL",
            "base_url": "RESEARCH_BASE_URL",
            "api_key": "RESEARCH_API_KEY",
        }
        values = {
            field_name: os.getenv(env_name, "").strip()
            for field_name, env_name in names.items()
        }
        missing = [names[field_name] for field_name, value in values.items() if not value]
        if missing:
            raise ConfigError(
                "Missing required research model configuration: " + ", ".join(missing)
            )
        values["base_url"] = values["base_url"].rstrip("/")
        langfuse_enabled = _bool_env("LANGFUSE_ENABLED", default=False)
        langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
        langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
        if langfuse_enabled and not (langfuse_public_key and langfuse_secret_key):
            raise ConfigError(
                "LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are required when "
                "LANGFUSE_ENABLED is true"
            )
        max_steps = _positive_int_env("RESEARCH_MAX_STEPS", default=12)
        return cls(
            **values,
            max_queries=_positive_int_env("RESEARCH_MAX_QUERIES", default=4),
            evidence_limit=_positive_int_env("RESEARCH_EVIDENCE_LIMIT", default=10),
            max_steps=max_steps,
            fast_max_steps=_positive_int_env(
                "RESEARCH_FAST_MAX_STEPS", default=min(max_steps, 8)
            ),
            worker_max_steps=_positive_int_env(
                "RESEARCH_WORKER_MAX_STEPS", default=30
            ),
            supervisor_max_steps=_positive_int_env(
                "RESEARCH_SUPERVISOR_MAX_STEPS", default=max_steps
            ),
            retrieval_top_k=_positive_int_env("RETRIEVAL_DEFAULT_TOP_K", default=8),
            max_evidence_reads=_positive_int_env("RESEARCH_MAX_EVIDENCE_READS", default=40),
            max_workers=_positive_int_env("RESEARCH_MAX_WORKERS", default=4),
            max_subtasks=_positive_int_env("RESEARCH_MAX_SUBTASKS", default=8),
            document_max_chars=_positive_int_env(
                "RESEARCH_DOCUMENT_MAX_CHARS", default=6000
            ),
            chapter_max_chars=_positive_int_env(
                "RESEARCH_CHAPTER_MAX_CHARS", default=1600
            ),
            chapter_max_claims=_positive_int_env(
                "RESEARCH_CHAPTER_MAX_CLAIMS", default=10
            ),
            chapter_max_decisions=_positive_int_env(
                "RESEARCH_CHAPTER_MAX_DECISIONS", default=4
            ),
            langfuse_enabled=langfuse_enabled,
            langfuse_public_key=langfuse_public_key,
            langfuse_secret_key=langfuse_secret_key,
            langfuse_base_url=os.getenv(
                "LANGFUSE_BASE_URL", "https://cloud.langfuse.com"
            ).strip().rstrip("/"),
        )


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

    # Only activates after repeated, large central orange GJB marks are confirmed.
    # Documents with no match continue through the original PDF prediction path.
    use_watermark_preprocessing: bool = True

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
