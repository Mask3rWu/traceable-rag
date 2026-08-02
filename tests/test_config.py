from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.config import ConfigError, DatabaseConfig, EmbeddingConfig, ParseConfig


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
