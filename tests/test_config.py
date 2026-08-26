from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.config import (
    ConfigError,
    DatabaseConfig,
    EmbeddingConfig,
    ParseConfig,
    ResearchModelConfig,
)


class ConfigTest(unittest.TestCase):
    def test_mobile_models_are_explicit(self):
        kwargs = ParseConfig().to_pipeline_kwargs()
        self.assertEqual(kwargs["layout_detection_model_name"], "PP-DocLayoutV3")
        self.assertEqual(kwargs["text_detection_model_name"], "PP-OCRv5_mobile_det")
        self.assertEqual(kwargs["text_recognition_model_name"], "PP-OCRv5_mobile_rec")
        self.assertEqual(kwargs["formula_recognition_model_name"], "PP-FormulaNet_plus-M")

    def test_embedding_config_loads_project_style_dotenv(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            env_file = Path(tmp) / ".env"
            env_file.write_text(
                "EMBEDDING_MODEL=test-model\n"
                "EMBEDDING_BASE_URL=https://embedding.example/v1/\n"
                "EMBEDDING_API_KEY=test-secret\n",
                encoding="utf-8",
            )

            config = EmbeddingConfig.from_env(env_file)

        self.assertEqual(config.model, "test-model")
        self.assertEqual(config.base_url, "https://embedding.example/v1")
        self.assertEqual(config.api_key, "test-secret")
        self.assertNotIn("test-secret", repr(config))

    def test_process_environment_overrides_dotenv(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text(
                "EMBEDDING_MODEL=file-model\n"
                "EMBEDDING_BASE_URL=https://file.example/v1\n"
                "EMBEDDING_API_KEY=file-secret\n",
                encoding="utf-8",
            )
            process_env = {
                "EMBEDDING_MODEL": "process-model",
                "EMBEDDING_BASE_URL": "https://process.example/v1",
                "EMBEDDING_API_KEY": "process-secret",
            }
            with patch.dict(os.environ, process_env, clear=True):
                config = EmbeddingConfig.from_env(env_file)

        self.assertEqual(config.model, "process-model")
        self.assertEqual(config.base_url, "https://process.example/v1")
        self.assertEqual(config.api_key, "process-secret")

    def test_embedding_config_reports_all_missing_values(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ConfigError) as raised:
                EmbeddingConfig.from_env(Path(tmp) / "missing.env")

        message = str(raised.exception)
        self.assertIn("EMBEDDING_MODEL", message)
        self.assertIn("EMBEDDING_BASE_URL", message)
        self.assertIn("EMBEDDING_API_KEY", message)

    def test_database_config_loads_dotenv_and_hides_password(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            env_file = Path(tmp) / ".env"
            env_file.write_text(
                "DB_HOST=127.0.0.1\n"
                "DB_PORT=5433\n"
                "DB_NAME=dba\n"
                "DB_USER=dba\n"
                "DB_PASSWORD=test-password\n",
                encoding="utf-8",
            )

            config = DatabaseConfig.from_env(env_file)

        self.assertEqual(config.host, "127.0.0.1")
        self.assertEqual(config.port, 5433)
        self.assertEqual(config.database, "dba")
        self.assertEqual(config.user, "dba")
        self.assertNotIn("test-password", repr(config))
        self.assertEqual(config.connection_kwargs()["dbname"], "dba")

    def test_database_config_rejects_invalid_port(self):
        env = {
            "DB_HOST": "127.0.0.1",
            "DB_PORT": "not-a-port",
            "DB_NAME": "dba",
            "DB_USER": "dba",
            "DB_PASSWORD": "test-password",
        }
        with patch.dict(os.environ, env, clear=True), self.assertRaises(ConfigError):
            DatabaseConfig.from_env()

    def test_research_model_config_loads_limits_and_hides_key(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            env_file = Path(tmp) / ".env"
            env_file.write_text(
                "RESEARCH_MODEL=test-model\n"
                "RESEARCH_BASE_URL=https://research.example/v1/\n"
                "RESEARCH_API_KEY=test-secret\n"
                "RESEARCH_MAX_QUERIES=3\n"
                "RESEARCH_EVIDENCE_LIMIT=7\n",
                encoding="utf-8",
            )

            config = ResearchModelConfig.from_env(env_file)

        self.assertEqual(config.base_url, "https://research.example/v1")
        self.assertEqual(config.max_queries, 3)
        self.assertEqual(config.evidence_limit, 7)
        self.assertEqual(config.fast_max_steps, 8)
        self.assertEqual(config.worker_max_steps, 30)
        self.assertEqual(config.supervisor_max_steps, 12)
        self.assertEqual(config.document_max_chars, 6000)
        self.assertEqual(config.chapter_max_chars, 1600)
        self.assertEqual(config.chapter_max_rules, 20)
        # 方案 A 旋钮默认关闭（维持现状，opt-in）
        self.assertFalse(config.disable_thinking_fast)
        self.assertFalse(config.disable_thinking_planner)
        self.assertFalse(config.disable_thinking_worker)
        self.assertFalse(config.disable_thinking_reviewer)
        self.assertNotIn("test-secret", repr(config))

    def test_research_model_config_thinking_knobs_from_env(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            env_file = Path(tmp) / ".env"
            env_file.write_text(
                "RESEARCH_MODEL=test-model\n"
                "RESEARCH_BASE_URL=https://research.example/v1/\n"
                "RESEARCH_API_KEY=test-secret\n"
                "RESEARCH_DISABLE_THINKING_FAST=1\n"
                "RESEARCH_DISABLE_THINKING_PLANNER=true\n"
                "RESEARCH_DISABLE_THINKING_WORKER=true\n"
                "RESEARCH_DISABLE_THINKING_REVIEWER=true\n",
                encoding="utf-8",
            )

            config = ResearchModelConfig.from_env(env_file)

        self.assertTrue(config.disable_thinking_fast)
        self.assertTrue(config.disable_thinking_planner)
        self.assertTrue(config.disable_thinking_worker)
        self.assertTrue(config.disable_thinking_reviewer)

    def test_research_model_config_token_budget_defaults_to_none(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            env_file = Path(tmp) / ".env"
            env_file.write_text(
                "RESEARCH_MODEL=test-model\n"
                "RESEARCH_BASE_URL=https://research.example/v1/\n"
                "RESEARCH_API_KEY=test-secret\n",
                encoding="utf-8",
            )

            config = ResearchModelConfig.from_env(env_file)

        # 方案 B 默认关闭（不限 token，经 env 留空触发；维持现状，opt-in）
        self.assertIsNone(config.fast_token_budget)
        self.assertIsNone(config.planner_token_budget)
        self.assertIsNone(config.worker_token_budget)
        self.assertIsNone(config.reviewer_token_budget)

    def test_research_model_config_token_budget_from_env(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            env_file = Path(tmp) / ".env"
            env_file.write_text(
                "RESEARCH_MODEL=test-model\n"
                "RESEARCH_BASE_URL=https://research.example/v1/\n"
                "RESEARCH_API_KEY=test-secret\n"
                "RESEARCH_FAST_TOKEN_BUDGET=1500\n"
                "RESEARCH_PLANNER_TOKEN_BUDGET=2000\n"
                "RESEARCH_WORKER_TOKEN_BUDGET=3000\n"
                "RESEARCH_REVIEWER_TOKEN_BUDGET=16000\n",
                encoding="utf-8",
            )

            config = ResearchModelConfig.from_env(env_file)

        self.assertEqual(config.fast_token_budget, 1500)
        self.assertEqual(config.planner_token_budget, 2000)
        self.assertEqual(config.worker_token_budget, 3000)
        self.assertEqual(config.reviewer_token_budget, 16000)

    def test_research_model_config_rejects_non_positive_token_budget(self):
        env = {
            "RESEARCH_MODEL": "test-model",
            "RESEARCH_BASE_URL": "https://research.example/v1",
            "RESEARCH_API_KEY": "test-secret",
            "RESEARCH_FAST_TOKEN_BUDGET": "0",
        }
        with patch.dict(os.environ, env, clear=True), self.assertRaises(ConfigError):
            ResearchModelConfig.from_env()
