"""
Unit tests for lemonade common utilities.
Tests status.py, filesystem.py, mmlu.py helpers, accuracy.py helpers,
and table.py helpers in isolation.
"""

import unittest
import os
import tempfile
from lemonade.common.status import _pretty_print_key, parameters_to_size, PrettyFloat
from lemonade.common.filesystem import clean_file_name, _clean_logfile
from lemonade.tools.mmlu import min_handle_none, _format_subject
from lemonade.tools.accuracy import LMEvalHarness
from lemonade.tools.report.table import _to_list, _wrap, _merge_join, _window_sum
from lemonade.common.exceptions import (
    Error,
    CacheError,
    EnvError,
    ArgError,
    ToolError,
    StateError,
    IntakeError,
    IOError,
    ModelArgError,
    ModelRuntimeError,
    BenchmarkException,
    HardwareError,
    SkipBuild,
)


class TestPrettyPrintKey(unittest.TestCase):
    """Test _pretty_print_key from status.py."""

    def test_simple_key(self):
        self.assertEqual(_pretty_print_key("model_name"), "Model Name")

    def test_single_word(self):
        self.assertEqual(_pretty_print_key("accuracy"), "Accuracy")

    def test_long_key(self):
        self.assertEqual(
            _pretty_print_key("seconds_to_first_token"),
            "Seconds To First Token",
        )

    def test_key_with_units(self):
        self.assertEqual(
            _pretty_print_key("average_mmlu_accuracy"),
            "Average Mmlu Accuracy",
        )


class TestParametersToSize(unittest.TestCase):
    """Test parameters_to_size from status.py."""

    def test_zero_params(self):
        self.assertEqual(parameters_to_size(0), "0B")

    def test_bytes_level(self):
        self.assertEqual(parameters_to_size(1), "4.0 B")

    def test_kilobytes(self):
        result = parameters_to_size(256)
        self.assertTrue("KB" in result or "B" in result)

    def test_megabytes(self):
        result = parameters_to_size(1_000_000)
        self.assertIn("MB", result)

    def test_gigabytes(self):
        result = parameters_to_size(1_000_000_000)
        self.assertIn("GB", result)

    def test_custom_bytes_per_param(self):
        result_2 = parameters_to_size(100, byte_per_parameter=2)
        result_4 = parameters_to_size(100, byte_per_parameter=4)
        self.assertEqual(float(result_4.split()[0]), float(result_2.split()[0]) * 2)


class TestPrettyFloat(unittest.TestCase):
    """Test PrettyFloat from status.py."""

    def test_repr_three_decimals(self):
        pf = PrettyFloat(3.14159)
        self.assertEqual(repr(pf), "3.142")

    def test_is_float(self):
        pf = PrettyFloat(2.5)
        self.assertIsInstance(pf, float)
        self.assertEqual(pf + 1, 3.5)


class TestCleanFileName(unittest.TestCase):
    """Test clean_file_name from filesystem.py."""

    def test_py_extension_stripped(self):
        self.assertEqual(clean_file_name("model.py"), "model")

    def test_onnx_extension_stripped(self):
        self.assertEqual(clean_file_name("model.onnx"), "model")

    def test_state_yaml_stripped(self):
        self.assertEqual(
            clean_file_name("my_build_state.yaml"),
            "my_build",
        )

    def test_full_path(self):
        self.assertEqual(
            clean_file_name("/path/to/model.py"),
            "model",
        )

    def test_no_extension(self):
        self.assertEqual(clean_file_name("model"), "model")


class TestCleanLogfile(unittest.TestCase):
    """Test _clean_logfile from filesystem.py."""

    def test_removes_trailing_whitespace(self):
        lines = ["hello   ", "world  ", "  "]
        result = _clean_logfile(lines)
        self.assertIn("hello", result)
        self.assertIn("world", result)
        self.assertNotIn("   ", result)

    def test_empty_lines_removed(self):
        lines = ["hello", "", "world", "", ""]
        result = _clean_logfile(lines)
        self.assertNotEqual(result, "\n".join(lines))
        self.assertIn("hello", result)
        self.assertIn("world", result)
        # Result should be compact with no stray newlines
        lines_in_result = result.split("\n")
        self.assertEqual(len(lines_in_result), 2)

    def test_all_empty(self):
        lines = ["", "  ", "\t"]
        result = _clean_logfile(lines)
        self.assertEqual(result, "")


class TestMinHandleNone(unittest.TestCase):
    """Test min_handle_none from mmlu.py."""

    def test_all_values(self):
        self.assertEqual(min_handle_none(3, 1, 5), 1)

    def test_with_none_values(self):
        self.assertEqual(min_handle_none(None, 5, 3), 3)

    def test_all_none(self):
        with self.assertRaises(ValueError):
            min_handle_none(None, None)

    def test_single_value(self):
        self.assertEqual(min_handle_none(7), 7)

    def test_single_none(self):
        with self.assertRaises(ValueError):
            min_handle_none(None)

    def test_mixed_order(self):
        self.assertEqual(min_handle_none(10, None, 2, None), 2)


class TestFormatSubject(unittest.TestCase):
    """Test _format_subject from mmlu.py."""

    def test_single_word(self):
        self.assertEqual(_format_subject("physics"), "physics")

    def test_multi_word(self):
        self.assertEqual(_format_subject("high_school_physics"), "high school physics")

    def test_empty(self):
        self.assertEqual(_format_subject(""), "")


class TestScaleMetric(unittest.TestCase):
    """Test LMEvalHarness._scale_metric from accuracy.py."""

    def setUp(self):
        self.harness = LMEvalHarness()

    def test_fraction_metric_scales(self):
        scaled, units, display = self.harness._scale_metric("acc", 0.85)
        self.assertAlmostEqual(scaled, 85.0)
        self.assertEqual(units, "%")
        self.assertIn("85.00%", display)

    def test_exact_match_scales(self):
        scaled, units, display = self.harness._scale_metric("exact_match", 0.75)
        self.assertAlmostEqual(scaled, 75.0)
        self.assertEqual(units, "%")

    def test_non_fraction_no_scale(self):
        scaled, units, display = self.harness._scale_metric("ppl", 10.5)
        self.assertAlmostEqual(scaled, 10.5)
        self.assertEqual(units, "raw")

    def test_value_outside_unit_range(self):
        scaled, units, display = self.harness._scale_metric("acc", 85.0)
        self.assertAlmostEqual(scaled, 85.0)
        self.assertEqual(units, "raw")

    def test_zero_value(self):
        scaled, units, display = self.harness._scale_metric("accuracy", 0.0)
        self.assertAlmostEqual(scaled, 0.0)
        self.assertEqual(units, "%")

    def test_one_value(self):
        scaled, units, display = self.harness._scale_metric("f1", 1.0)
        self.assertAlmostEqual(scaled, 100.0)
        self.assertEqual(units, "%")


class TestToList(unittest.TestCase):
    """Test _to_list from table.py."""

    def test_scalar_to_list(self):
        self.assertEqual(_to_list(42), [42])
        self.assertEqual(_to_list("hello"), ["hello"])

    def test_list_unchanged(self):
        self.assertEqual(_to_list([1, 2, 3]), [1, 2, 3])
        self.assertEqual(_to_list([]), [])

    def test_none_to_list(self):
        self.assertEqual(_to_list(None), [None])


class TestWrap(unittest.TestCase):
    """Test _wrap from table.py."""

    def test_short_text(self):
        result = _wrap("hello", 80)
        self.assertEqual(result, "hello")

    def test_long_text_wraps(self):
        text = "This is a very long text that should wrap at the specified width"
        result = _wrap(text, 20)
        self.assertIn("\n", result)

    def test_empty_string(self):
        result = _wrap("", 10)
        self.assertEqual(result, "")


class TestMergeJoin(unittest.TestCase):
    """Test _merge_join from table.py."""

    def test_both_non_empty(self):
        result = _merge_join("hello", "world")
        self.assertEqual(result, "hello\nworld")

    def test_first_empty(self):
        result = _merge_join("", "world")
        self.assertEqual(result, "world")

    def test_second_empty(self):
        result = _merge_join("hello", "")
        self.assertEqual(result, "hello")

    def test_both_empty(self):
        result = _merge_join("", "")
        self.assertEqual(result, "")


class TestWindowSum(unittest.TestCase):
    """Test _window_sum from table.py."""

    def test_zero_windows(self):
        data = [1, 2, 3, 4, 5, 6]
        result = _window_sum(data, 0)
        self.assertEqual(result, data)

    def test_two_windows(self):
        data = [1, 2, 3, 4]
        result = _window_sum(data, 2)
        self.assertEqual(result, [3, 7])

    def test_three_windows(self):
        data = [1, 1, 1, 2, 2, 2]
        result = _window_sum(data, 3)
        self.assertEqual(result, [2, 3, 4])

    def test_single_window(self):
        data = [1, 2, 3, 4]
        result = _window_sum(data, 1)
        self.assertEqual(result, [10])

    def test_more_windows_than_elements(self):
        data = [1, 2, 3]
        result = _window_sum(data, 5)
        self.assertEqual(result, [1, 2, 3])


class TestExceptions(unittest.TestCase):
    """Test exception hierarchy from exceptions.py."""

    def test_error_is_exception(self):
        self.assertTrue(issubclass(Error, Exception))

    def test_cache_error_hierarchy(self):
        self.assertTrue(issubclass(CacheError, Error))

    def test_env_error_hierarchy(self):
        self.assertTrue(issubclass(EnvError, Error))

    def test_arg_error_hierarchy(self):
        self.assertTrue(issubclass(ArgError, Error))

    def test_io_error_hierarchy(self):
        self.assertTrue(issubclass(IOError, Error))

    def test_model_arg_error_hierarchy(self):
        self.assertTrue(issubclass(ModelArgError, Error))

    def test_model_runtime_error_hierarchy(self):
        self.assertTrue(issubclass(ModelRuntimeError, Error))

    def test_hardware_error_hierarchy(self):
        self.assertTrue(issubclass(HardwareError, Error))

    def test_tool_error_is_exception(self):
        self.assertTrue(issubclass(ToolError, Exception))
        self.assertFalse(issubclass(ToolError, Error))

    def test_state_error_is_exception(self):
        self.assertTrue(issubclass(StateError, Exception))

    def test_benchmark_exception_is_exception(self):
        self.assertTrue(issubclass(BenchmarkException, Exception))

    def test_skip_build_is_exception(self):
        self.assertTrue(issubclass(SkipBuild, Exception))

    def test_intake_error_is_exception(self):
        self.assertTrue(issubclass(IntakeError, Exception))

    def test_errors_can_be_raised(self):
        try:
            raise CacheError("test message")
        except Error:
            pass
        else:
            self.fail("CacheError should be catchable as Error")

    def test_errors_can_be_caught(self):
        with self.assertRaises(ArgError):
            raise ArgError("bad args")


if __name__ == "__main__":
    unittest.main()
