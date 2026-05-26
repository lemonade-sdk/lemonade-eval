"""
Unit tests for lemonade.cache module.
Tests cache utility functions in isolation with no external dependencies.
"""

import unittest
import os
import tempfile
from datetime import datetime
from lemonade.cache import (
    checkpoint_to_model_name,
    get_build_timestamp,
    build_name,
    Keys,
    DEFAULT_CACHE_DIR,
)


class TestCheckpointToModelName(unittest.TestCase):
    def test_standard_checkpoint(self):
        self.assertEqual(checkpoint_to_model_name("author/model"), "model")

    def test_org_checkpoint(self):
        self.assertEqual(checkpoint_to_model_name("org/model-name"), "model-name")

    def test_single_part_raises(self):
        with self.assertRaises(IndexError):
            checkpoint_to_model_name("justauthorname")

    def test_multiple_slashes(self):
        self.assertEqual(checkpoint_to_model_name("a/b/c"), "b")


class TestGetBuildTimestamp(unittest.TestCase):
    def test_format(self):
        dt = datetime(2025, 1, 15, 13, 30, 45)
        ts = get_build_timestamp(dt)
        self.assertEqual(ts, "2025y_01m_15d_13h_30m_45s")

    def test_midnight(self):
        dt = datetime(2025, 12, 1, 0, 0, 0)
        ts = get_build_timestamp(dt)
        self.assertEqual(ts, "2025y_12m_01d_00h_00m_00s")

    def test_end_of_year(self):
        dt = datetime(2025, 12, 31, 23, 59, 59)
        ts = get_build_timestamp(dt)
        self.assertEqual(ts, "2025y_12m_31d_23h_59m_59s")


class TestBuildName(unittest.TestCase):
    def setUp(self):
        self.build_time = datetime(2025, 6, 15, 12, 0, 0)

    def test_remote_checkpoint(self):
        name = build_name("org/model-name", self.build_time)
        self.assertTrue(name.startswith("org_model-name_"))
        self.assertIn("2025y_06m_15d_12h_00m_00s", name)

    def test_checkpoint_with_colon(self):
        name = build_name("org/model:v1", self.build_time)
        self.assertTrue(name.startswith("org_model-v1_"))

    def test_local_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            name = build_name(tmpdir, self.build_time)
            self.assertTrue(name.startswith("local_model_"))

    def test_local_file(self):
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
            f.write(b"test")
            filepath = f.name
        try:
            name = build_name(filepath, self.build_time)
            expected_prefix = os.path.splitext(os.path.basename(filepath))[0]
            self.assertTrue(name.startswith(f"{expected_prefix}_"))
        finally:
            os.unlink(filepath)

    def test_simple_name(self):
        name = build_name("simplemodel", self.build_time)
        self.assertTrue(name.startswith("simplemodel_"))
        self.assertTrue(name.endswith("2025y_06m_15d_12h_00m_00s"))


class TestKeys(unittest.TestCase):
    def test_keys_are_strings(self):
        for attr in dir(Keys):
            if not attr.startswith("_"):
                self.assertIsInstance(getattr(Keys, attr), str)

    def test_known_keys_present(self):
        self.assertEqual(Keys.MODEL, "model")
        self.assertEqual(Keys.SECONDS_TO_FIRST_TOKEN, "seconds_to_first_token")
        self.assertEqual(Keys.TOKEN_GENERATION_TOKENS_PER_SECOND, "token_generation_tokens_per_second")
        self.assertEqual(Keys.CHECKPOINT, "checkpoint")
        self.assertEqual(Keys.PROMPT, "prompt")
        self.assertEqual(Keys.RESPONSE, "response")


class TestDefaultCacheDir(unittest.TestCase):
    def test_default_is_home_cache(self):
        self.assertIn(".cache", DEFAULT_CACHE_DIR)
        self.assertIn("lemonade", DEFAULT_CACHE_DIR)


if __name__ == "__main__":
    unittest.main()
