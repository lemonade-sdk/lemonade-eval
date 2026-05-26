"""
Unit tests for lemonade.state module.
Tests State class and YAML sanitization helpers in isolation.
"""

import unittest
import os
import tempfile
from datetime import datetime
from lemonade.state import _is_nice_to_write, _sanitize_for_yaml, State


class TestIsNiceToWrite(unittest.TestCase):
    def test_primitives_are_nice(self):
        self.assertTrue(_is_nice_to_write("hello"))
        self.assertTrue(_is_nice_to_write(42))
        self.assertTrue(_is_nice_to_write(3.14))
        self.assertTrue(_is_nice_to_write(True))
        self.assertTrue(_is_nice_to_write(False))

    def test_list_of_primitives_are_nice(self):
        self.assertTrue(_is_nice_to_write(["a", "b", "c"]))
        self.assertTrue(_is_nice_to_write([1, 2, 3]))
        self.assertTrue(_is_nice_to_write([]))

    def test_tuple_are_nice(self):
        self.assertTrue(_is_nice_to_write((1, 2, 3)))

    def test_dict_of_primitives_are_nice(self):
        self.assertTrue(_is_nice_to_write({"a": 1, "b": "x"}))
        self.assertTrue(_is_nice_to_write({}))

    def test_nested_structures_are_nice(self):
        self.assertTrue(_is_nice_to_write({"a": [1, 2], "b": {"c": "d"}}))

    def test_none_is_not_nice(self):
        self.assertFalse(_is_nice_to_write(None))

    def test_objects_are_not_nice(self):
        self.assertFalse(_is_nice_to_write(object()))
        self.assertFalse(_is_nice_to_write(datetime.now()))

    def test_list_with_none_is_not_nice(self):
        self.assertFalse(_is_nice_to_write([1, None, 3]))

    def test_dict_with_none_is_not_nice(self):
        self.assertFalse(_is_nice_to_write({"a": None}))


class TestSanitizeForYaml(unittest.TestCase):
    def test_filters_none_values(self):
        result = _sanitize_for_yaml({"a": "keep", "b": None, "c": 42})
        self.assertEqual(result, {"a": "keep", "c": 42})

    def test_keeps_all_good_values(self):
        result = _sanitize_for_yaml({"a": "x", "b": 1, "c": 3.0, "d": True, "e": ["y"], "f": {"z": 2}})
        self.assertEqual(len(result), 6)

    def test_empty_dict(self):
        result = _sanitize_for_yaml({})
        self.assertEqual(result, {})

    def test_all_bad_values(self):
        result = _sanitize_for_yaml({"a": None, "b": object()})
        self.assertEqual(result, {})


class TestStateInit(unittest.TestCase):
    def test_init_sets_defaults(self):
        state = State(cache_dir="/tmp/test_cache")
        self.assertEqual(state.cache_dir, "/tmp/test_cache")
        self.assertIsNotNone(state.build_name)
        self.assertIsNotNone(state.build_time)
        self.assertIsNotNone(state.uid)
        self.assertIsNotNone(state.build_status)

    def test_init_with_build_name(self):
        state = State(cache_dir="/tmp/test_cache", build_name="my_build")
        self.assertEqual(state.build_name, "my_build")

    def test_init_with_build_time(self):
        dt = datetime(2025, 1, 1)
        state = State(cache_dir="/tmp/test_cache", build_time=dt)
        self.assertEqual(state.build_time, dt)

    def test_init_with_sequence_info(self):
        seq = {"tool1": {"args": {"a": 1}}}
        state = State(cache_dir="/tmp/test_cache", sequence_info=seq)
        self.assertEqual(state.sequence_info, seq)

    def test_init_expands_user(self):
        state = State(cache_dir="~/lemonade_cache")
        self.assertNotIn("~", state.cache_dir)

    def test_init_extra_kwargs(self):
        state = State(cache_dir="/tmp/test_cache", custom_attr="hello")
        self.assertEqual(state.custom_attr, "hello")

    def test_init_sets_version(self):
        state = State(cache_dir="/tmp/test_cache")
        self.assertTrue(hasattr(state, "lemonade_version"))

    def test_init_sets_downcast(self):
        state = State(cache_dir="/tmp/test_cache")
        self.assertFalse(state.downcast_applied)


class TestStateSetAttr(unittest.TestCase):
    def test_can_set_arbitrary_attr(self):
        state = State(cache_dir="/tmp/test_cache")
        state.new_attr = "value"
        self.assertEqual(state.new_attr, "value")
        state.model = object()
        self.assertIsNotNone(state.model)


class TestStateSaveMethods(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_save_stores_yaml(self):
        state = State(cache_dir=self.tmpdir, build_name="test_build")
        state.test_key = "test_value"
        state.save()

        state_file = os.path.join(self.tmpdir, "builds", "test_build", "state.yaml")
        self.assertTrue(os.path.exists(state_file))

    def test_save_skips_non_yaml_types(self):
        state = State(cache_dir=self.tmpdir, build_name="test_build")
        state.good_attr = "saved"
        state.bad_attr = object()
        state.save()

        state_file = os.path.join(self.tmpdir, "builds", "test_build", "state.yaml")
        self.assertTrue(os.path.exists(state_file))

        import yaml
        with open(state_file, "r") as f:
            data = yaml.safe_load(f)
        self.assertIn("good_attr", data)
        self.assertNotIn("bad_attr", data)


if __name__ == "__main__":
    unittest.main()
