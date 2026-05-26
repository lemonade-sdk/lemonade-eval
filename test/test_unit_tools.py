"""
Unit tests for lemonade.tools modules.
Tests tool validation, prompt utilities, bench utilities,
and other tool logic in isolation.
"""

import unittest
import argparse
from lemonade.tools.tool import _name_is_file_safe, ToolParser, Tool, FirstTool
from lemonade.tools.prompt import sanitize_string, sanitize_text, positive_int
from lemonade.tools.bench import Bench, default_iterations, default_warmup_runs
from lemonade.tools.server_load import ServerAdapter


class TestNameIsFileSafe(unittest.TestCase):
    """Test the _name_is_file_safe validation function."""

    def test_valid_names(self):
        should_pass = [
            "load",
            "server-bench",
            "accuracy_mmlu",
            "a",
            "My-Tool_123",
            "llm-prompt",
            "123abc",
        ]
        for name in should_pass:
            try:
                _name_is_file_safe(name)
            except ValueError as e:
                self.fail(f"_name_is_file_safe('{name}') raised ValueError: {e}")

    def test_empty_name_raises(self):
        with self.assertRaises(ValueError):
            _name_is_file_safe("")

    def test_spaces_raise(self):
        with self.assertRaises(ValueError):
            _name_is_file_safe("my tool")

    def test_special_chars_raise(self):
        bad_names = [
            "tool/name",
            "tool.name",
            "tool name",
            "tool!name",
            "tool@name",
            "tool#name",
            "tool$name",
            "tool%name",
            "tool^name",
            "tool&name",
            "tool*name",
            "tool(name",
        ]
        for name in bad_names:
            with self.assertRaises(ValueError, msg=f"Name '{name}' should be invalid"):
                _name_is_file_safe(name)


class TestToolParser(unittest.TestCase):
    """Test the ToolParser argument parser."""

    def test_parser_creation(self):
        parser = ToolParser(
            short_description="Test tool",
            description="A test tool for unit tests",
            prog="lemonade test-tool",
            epilog="End of help",
        )
        self.assertEqual(parser.short_description, "Test tool")

    def test_parser_adds_arguments(self):
        parser = ToolParser(
            short_description="Test tool",
            description="A test tool",
            prog="lemonade test-tool",
            epilog="End of help",
        )
        parser.add_argument("--test", type=str, default="value")
        args = parser.parse_args(["--test", "hello"])
        self.assertEqual(args.test, "hello")


class TestSanitizeString(unittest.TestCase):
    """Test the sanitize_string function from prompt.py."""

    def test_ascii_string(self):
        result = sanitize_string("hello world")
        self.assertEqual(result, "hello world")

    def test_string_with_special_chars(self):
        result = sanitize_string("test\x00\x01\x02")
        self.assertTrue(result.startswith("test"), f"Expected 'test...' got {result!r}")

    def test_empty_string(self):
        result = sanitize_string("")
        self.assertEqual(result, "")

    def test_unicode(self):
        result = sanitize_string("caf\u00e9")
        self.assertEqual(result, "caf\u00e9")


class TestSanitizeText(unittest.TestCase):
    """Test the sanitize_text function from prompt.py."""

    def test_string_input(self):
        result = sanitize_text("hello world")
        self.assertEqual(result, "hello world")

    def test_list_of_strings(self):
        result = sanitize_text(["hello", "world"])
        self.assertEqual(result, ["hello", "world"])

    def test_non_string_raises(self):
        with self.assertRaises(TypeError):
            sanitize_text(42)
        with self.assertRaises(TypeError):
            sanitize_text(None)


class TestPositiveInt(unittest.TestCase):
    """Test the positive_int argparse type validator."""

    def test_valid_positive(self):
        self.assertEqual(positive_int("1"), 1)
        self.assertEqual(positive_int("42"), 42)
        self.assertEqual(positive_int("100"), 100)

    def test_zero_raises(self):
        with self.assertRaises(ValueError):
            positive_int("0")

    def test_negative_raises(self):
        with self.assertRaises(ValueError):
            positive_int("-1")
        with self.assertRaises(ValueError):
            positive_int("-5")

    def test_non_integer_raises(self):
        with self.assertRaises(ValueError):
            positive_int("abc")
        with self.assertRaises(ValueError):
            positive_int("3.14")


class TestBenchGetItemOrList(unittest.TestCase):
    """Test Bench.get_item_or_list static method."""

    def test_single_item_returns_item(self):
        self.assertEqual(Bench.get_item_or_list([42]), 42)
        self.assertEqual(Bench.get_item_or_list(["hello"]), "hello")
        self.assertEqual(Bench.get_item_or_list([None]), None)

    def test_multiple_items_returns_list(self):
        result = Bench.get_item_or_list([1, 2, 3])
        self.assertEqual(result, [1, 2, 3])

    def test_empty_list_returns_empty(self):
        self.assertEqual(Bench.get_item_or_list([]), [])


class TestBenchNotEnoughTokens(unittest.TestCase):
    """Test Bench.not_enough_tokens error message."""

    def test_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            Bench.not_enough_tokens(32)
        self.assertIn("32", str(ctx.exception))
        self.assertIn("output tokens", str(ctx.exception))

    def test_message_contains_recommendations(self):
        with self.assertRaises(ValueError) as ctx:
            Bench.not_enough_tokens(64)
        msg = str(ctx.exception)
        self.assertIn("1.", msg)


class TestParseImageSize(unittest.TestCase):
    """Test ServerAdapter._parse_image_size static method."""

    def test_exact_dimensions(self):
        self.assertEqual(
            ServerAdapter._parse_image_size("1024x800"), (1024, 800)
        )
        self.assertEqual(
            ServerAdapter._parse_image_size("1920X1080"), (1920, 1080)
        )

    def test_single_integer(self):
        self.assertEqual(
            ServerAdapter._parse_image_size("384"), (384, None)
        )
        self.assertEqual(
            ServerAdapter._parse_image_size("512"), (512, None)
        )

    def test_negative_dimension_raises(self):
        with self.assertRaises(ValueError):
            ServerAdapter._parse_image_size("-100x200")
        with self.assertRaises(ValueError):
            ServerAdapter._parse_image_size("100x-200")
        with self.assertRaises(ValueError):
            ServerAdapter._parse_image_size("-100")

    def test_zero_dimension_raises(self):
        with self.assertRaises(ValueError):
            ServerAdapter._parse_image_size("0x200")
        with self.assertRaises(ValueError):
            ServerAdapter._parse_image_size("200x0")
        with self.assertRaises(ValueError):
            ServerAdapter._parse_image_size("0")

    def test_invalid_format_raises(self):
        with self.assertRaises(ValueError):
            ServerAdapter._parse_image_size("abc")
        with self.assertRaises(ValueError):
            ServerAdapter._parse_image_size("1024xx800")
        with self.assertRaises(ValueError):
            ServerAdapter._parse_image_size("abcxdef")
        with self.assertRaises(ValueError):
            ServerAdapter._parse_image_size("")


if __name__ == "__main__":
    unittest.main()
