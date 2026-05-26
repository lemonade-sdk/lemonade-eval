"""
Unit tests for lemonade.common.cli_helpers module.
Tests CLI argument parsing in isolation.

Note: parse_tools() reads sys.argv, so these tests manipulate sys.argv
to exercise the parsing logic.
"""

import unittest
import sys
import argparse
from lemonade.common.cli_helpers import CustomArgumentParser, parse_tools
from lemonade.tools.tool import Tool
from lemonade.tools.management_tools import ManagementTool
from lemonade.state import State


class MockFirstTool(Tool):
    """Minimal FirstTool subclass for testing."""
    unique_name = "mock-first-tool"

    def __init__(self):
        super().__init__(monitor_message="Mocking first tool")
        self.monitor_message = "Mocking first tool"

    def run(self, state: State, input=None) -> State:
        return state

    @staticmethod
    def parser():
        parser = MockFirstTool.helpful_parser(
            short_description="A mock first tool for testing"
        )
        parser.add_argument("--input", default="default_input")
        return parser


class MockEvalTool(Tool):
    """Minimal EvalTool subclass for testing."""
    unique_name = "mock-eval-tool"

    def __init__(self):
        super().__init__(monitor_message="Mocking eval tool")
        self.monitor_message = "Mocking eval tool"

    def run(self, state: State) -> State:
        return state

    @staticmethod
    def parser():
        parser = MockEvalTool.helpful_parser(
            short_description="A mock eval tool for testing"
        )
        parser.add_argument("--option", default="default")
        return parser


class MockEvalTool2(Tool):
    """Second minimal EvalTool for multi-tool tests."""
    unique_name = "mock-eval-tool2"

    def __init__(self):
        super().__init__(monitor_message="Mocking eval tool 2")
        self.monitor_message = "Mocking eval tool 2"

    def run(self, state: State) -> State:
        return state

    @staticmethod
    def parser():
        parser = MockEvalTool2.helpful_parser(
            short_description="A second mock eval tool for testing"
        )
        parser.add_argument("--flag", action="store_true")
        return parser


class MockMgmtTool(ManagementTool):
    """Minimal ManagementTool subclass for testing."""
    unique_name = "mock-mgmt-tool"

    @staticmethod
    def parser():
        parser = MockMgmtTool.helpful_parser(
            short_description="A mock management tool for testing"
        )
        parser.add_argument("--action", default="list")
        return parser

    def run(self, cache_dir: str, action="list"):
        pass


class TestCustomArgumentParser(unittest.TestCase):
    """Test the CustomArgumentParser class (overrides error method)."""

    def test_parser_creation(self):
        parser = CustomArgumentParser()
        self.assertIsInstance(parser, argparse.ArgumentParser)

    def test_parser_add_argument(self):
        parser = CustomArgumentParser()
        parser.add_argument("--test", type=str)
        args = parser.parse_args(["--test", "value"])
        self.assertEqual(args.test, "value")


class TestParseToolsBasic(unittest.TestCase):
    """Test parse_tools basic scenarios."""

    def setUp(self):
        self.supported_tools = [MockFirstTool, MockEvalTool, MockEvalTool2, MockMgmtTool]
        self.parser = CustomArgumentParser()
        self.parser.add_argument("-i", "--input", type=str)
        self.parser.add_argument("-d", "--cache-dir", type=str, default="/tmp/test_cache")

    def tearDown(self):
        sys.argv = [sys.argv[0]]

    def test_single_eval_tool(self):
        sys.argv = [
            "lemonade-eval",
            "-i", "my_model",
            "mock-first-tool",
            "mock-eval-tool",
        ]
        global_args, tool_instances, eval_tools = parse_tools(
            self.parser, self.supported_tools
        )
        self.assertIn("input", global_args)
        self.assertIn("cache_dir", global_args)
        tool_names = [t.__class__.unique_name for t in tool_instances.keys()]
        self.assertIn("mock-first-tool", tool_names)
        self.assertIn("mock-eval-tool", tool_names)
        self.assertEqual(len(eval_tools), 2)

    def test_single_mgmt_tool(self):
        sys.argv = [
            "lemonade-eval",
            "mock-mgmt-tool",
            "--action", "list",
        ]
        global_args, tool_instances, eval_tools = parse_tools(
            self.parser, self.supported_tools
        )
        tool_names = [t.__class__.unique_name for t in tool_instances.keys()]
        self.assertIn("mock-mgmt-tool", tool_names)
        self.assertEqual(len(eval_tools), 0)

    def test_multi_tool_sequence(self):
        sys.argv = [
            "lemonade-eval",
            "-i", "model_name",
            "mock-first-tool",
            "--input", "custom",
            "mock-eval-tool",
            "--option", "fast",
            "mock-eval-tool2",
            "--flag",
        ]
        global_args, tool_instances, eval_tools = parse_tools(
            self.parser, self.supported_tools
        )
        self.assertEqual(global_args["input"], "model_name")
        self.assertEqual(len(tool_instances), 3)
        self.assertEqual(len(eval_tools), 3)

    def test_no_tools_raises_error(self):
        sys.argv = [
            "lemonade-eval",
            "-i", "model_name",
        ]
        self.parser = CustomArgumentParser()
        self.parser.add_argument("-i", "--input", type=str)
        with self.assertRaises(SystemExit):
            parse_tools(self.parser, self.supported_tools)


class TestParseToolsAdvanced(unittest.TestCase):
    """Test parse_tools edge cases."""

    def setUp(self):
        self.supported_tools = [MockFirstTool, MockEvalTool, MockMgmtTool]

    def tearDown(self):
        sys.argv = [sys.argv[0]]

    def test_duplicate_tool_rejected(self):
        sys.argv = [
            "lemonade-eval",
            "mock-eval-tool",
            "mock-eval-tool",
        ]
        parser = CustomArgumentParser()
        parser.add_argument("-i", "--input", type=str)
        with self.assertRaises(SystemExit):
            parse_tools(parser, self.supported_tools)

    def test_mixed_mgmt_and_eval_rejected(self):
        sys.argv = [
            "lemonade-eval",
            "mock-first-tool",
            "mock-mgmt-tool",
        ]
        parser = CustomArgumentParser()
        parser.add_argument("-i", "--input", type=str)
        with self.assertRaises(SystemExit):
            parse_tools(parser, self.supported_tools)


if __name__ == "__main__":
    unittest.main()
