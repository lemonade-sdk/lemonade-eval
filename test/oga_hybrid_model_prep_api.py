import unittest
import shutil
import os
from lemonade.state import State
import lemonade.common.test_helpers as common
from lemonade.common.build import builds_dir
from lemonade.tools.prompt import LLMPrompt
from lemonade.tools.oga.load import OgaLoad
import sys

ci_mode = os.getenv("LEMONADE_CI_MODE", False)

checkpoint = "amd/Llama-3.2-1B-Instruct-awq-uint4-asym-g128-bf16-lmhead"
device = "hybrid"
dtype = "int4"
force = False
prompt = "Alice and Bob"


class Testing(unittest.TestCase):

    def setUp(self) -> None:
        shutil.rmtree(builds_dir(cache_dir), ignore_errors=True)

    def test_001_oga_model_prep_hybrid(self):
        # Test the OgaLoad with model generation (oga_model_prep) for hybrid device
        # and LLMPrompt tools

        state = State(cache_dir=cache_dir, build_name="test")

        state = OgaLoad().run(
            state,
            input=checkpoint,
            device=device,
            dtype=dtype,
            force=force,
            dml_only=True,
        )
        state = LLMPrompt().run(state, prompt=prompt, max_new_tokens=10)

        assert len(state.response) > 0, state.response


if __name__ == "__main__":
    cache_dir, _ = common.create_test_dir(
        "lemonade_oga_hybrid_model_prep_api", base_dir=os.path.abspath(".")
    )

    suite = unittest.TestSuite()
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(Testing))

    # Run the test suite
    runner = unittest.TextTestRunner()
    result = runner.run(suite)

    # Set exit code based on test results
    if not result.wasSuccessful():
        sys.exit(1)

# This file was originally licensed under Apache 2.0. It has been modified.
# Modifications Copyright (c) 2025 AMD
