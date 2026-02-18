"""
Tool for benchmarking a model loaded on Lemonade Server.
"""

import argparse
import statistics
from statistics import StatisticsError

from lemonade.state import State
from lemonade.tools.bench import Bench
from lemonade.tools.server_load import ServerAdapter


class ServerBench(Bench):
    """
    Benchmark a model loaded on Lemonade Server.

    This tool uses the /api/v1/chat/completions endpoint for inference and the
    /api/v1/stats endpoint to collect performance metrics. It follows the
    same benchmarking methodology as other *-bench tools.

    Supports both text-only LLMs and Vision-Language Models (VLMs). For VLMs,
    use the --image flag to provide an image file or URL.

    Required input state:
        - model: ServerAdapter instance (set by the `load` tool)
        - tokenizer: ServerTokenizerAdapter instance

    Output state produced:
        - Performance statistics including TTFT, tokens/second, etc.

    Example usage:
        lemonade-eval -i Qwen3-0.6B-GGUF load bench
        lemonade-eval -i Qwen3-4B-VL-FLM load bench --image photo.jpg
    """

    unique_name = "bench"

    def __init__(self):
        super().__init__(monitor_message="Benchmarking model on Lemonade Server")
        self._image = None

    @staticmethod
    def parser(add_help: bool = True) -> argparse.ArgumentParser:
        parser = __class__.helpful_parser(
            short_description="Benchmark a model on Lemonade Server",
            add_help=add_help,
        )

        parser = Bench.parser(parser)

        parser.add_argument(
            "--image",
            type=str,
            default=None,
            help="Path to an image file or URL for VLM (Vision-Language Model) "
            "benchmarking. When provided, each benchmark iteration sends a "
            "multimodal prompt containing both the image and text. "
            "The -p flag controls the text portion of the prompt.",
        )

        parser.add_argument(
            "--image-size",
            type=str,
            default=None,
            help="Resize the image before sending to the server. Accepts "
            "WIDTHxHEIGHT (e.g. --image-size 1024x800) to resize to exact "
            "dimensions, or a single integer (e.g. --image-size 384) to cap "
            "the longest side while preserving aspect ratio. Reduces visual "
            "token count for VLM models. Only applies when --image is set.",
        )

        return parser

    # Prefix to encourage long model responses for benchmarking
    PROMPT_PREFIX = (
        "Tell me an extremely long story that starts with the following "
        "but goes from there: "
    )

    # Prefix for VLM benchmarking that encourages long responses about the image
    VLM_PROMPT_PREFIX = (
        "Describe this image in extreme detail, covering every single element, "
        "color, texture, shape, and spatial relationship you can observe. "
        "Then tell an extremely long creative story inspired by the image: "
    )

    def parse(self, state, args, known_only=True):
        """
        Override parse to extract --image before prompt processing, so that
        get_prompt_str() can use a VLM-appropriate prompt prefix.
        """
        pre_parser = argparse.ArgumentParser(add_help=False)
        pre_parser.add_argument("--image", type=str, default=None)
        pre_parser.add_argument("--image-size", type=str, default=None)
        pre_args, _ = pre_parser.parse_known_args(args)
        self._image = pre_args.image

        return super().parse(state, args, known_only)

    def get_prompt_str(self, state, token_length):
        """
        Returns a string with approximately the prescribed token length.

        The prompt includes a prefix that encourages long responses, followed
        by synthetic "word" tokens. We use calibration via the server's actual
        token count to match the target length.

        For VLM models (when --image is set), uses a VLM-appropriate prefix
        that encourages detailed image description. The -p token count controls
        only the text portion; image tokens are additional.
        """
        model: ServerAdapter = state.model

        # Choose prefix based on whether this is a VLM benchmark
        prefix = self.VLM_PROMPT_PREFIX if self._image else self.PROMPT_PREFIX

        # Start with an initial estimate: prefix + "word " repeated
        # Assume prefix is ~20 tokens, so start with (token_length - 20) words
        initial_word_count = max(1, token_length - 20)
        test_prompt = prefix + "word " * initial_word_count

        # Make a calibration request to get the actual token count
        try:
            _ = model.generate(test_prompt, max_new_tokens=1)
            actual_tokens = model.prompt_tokens

            if actual_tokens is None:
                # Fall back to estimation if stats unavailable
                return test_prompt

            # Adjust based on the difference
            delta = actual_tokens - token_length
            if delta == 0:
                return test_prompt

            # Calculate adjusted word count
            adjusted_words = max(1, initial_word_count - delta)
            return prefix + "word " * adjusted_words

        except Exception:  # pylint: disable=broad-exception-caught
            # If calibration fails, use initial estimation
            return test_prompt

    def run_prompt(
        self,
        state: State,
        report_progress_fn,
        prompt: str,
        iterations: int,
        warmup_iterations: int,
        output_tokens: int,
        **kwargs,
    ):
        """
        Benchmark the server model for a single prompt configuration.

        Args:
            state: Lemonade state containing the ServerAdapter model
            report_progress_fn: Callback to report progress
            prompt: Input prompt text
            iterations: Number of benchmark iterations
            warmup_iterations: Number of warmup iterations (not counted in results)
            output_tokens: Target number of output tokens
            **kwargs: Additional arguments, including 'image' for VLM benchmarking
        """
        image = kwargs.get("image", None)
        image_size = kwargs.get("image_size", None)

        # Prepare the image URL once to avoid redundant disk I/O, resizing,
        # and base64 encoding on every iteration.
        if image is not None:
            # pylint: disable-next=protected-access
            image = ServerAdapter._prepare_image_url(image, image_size=image_size)
            image_size = None

        if self.first_run_prompt:
            if not hasattr(state, "model"):
                raise ValueError(
                    f"{self.__class__.unique_name} requires a model to be loaded first. "
                    "Please run the 'load' tool before this tool."
                )

            if not isinstance(state.model, ServerAdapter):
                raise ValueError(
                    f"{self.__class__.unique_name} requires a ServerAdapter model. "
                    f"Got {type(state.model).__name__} instead. "
                    "Please use the 'load' tool to load a model on Lemonade Server."
                )

        model: ServerAdapter = state.model

        per_iteration_time_to_first_token = []
        per_iteration_tokens_per_second = []
        per_iteration_peak_wset = []

        total_iterations = warmup_iterations + iterations

        for iteration in range(total_iterations):
            try:
                # Reset metrics before each call
                model.time_to_first_token = None
                model.tokens_per_second = None
                model.prompt_tokens = None
                model.response_tokens = None
                model.peak_wset = None

                # Run inference with memory tracking if enabled
                _ = model.generate(
                    prompt,
                    max_new_tokens=output_tokens,
                    save_max_memory_used=self.save_max_memory_used,
                    image=image,
                    image_size=image_size,
                )

                # Check that we got valid metrics
                if model.time_to_first_token is None or model.tokens_per_second is None:
                    raise RuntimeError(
                        "Could not retrieve timing information from server. "
                        "Make sure the server supports the /api/v1/stats endpoint."
                    )

                # Record output tokens for all iterations
                if model.response_tokens is not None:
                    self.tokens_out_len_list.append(model.response_tokens)

                # Only record metrics after warmup
                if iteration >= warmup_iterations:
                    # Only count if we got enough output tokens
                    if (
                        model.response_tokens is not None
                        and model.response_tokens >= output_tokens
                    ):
                        per_iteration_time_to_first_token.append(
                            model.time_to_first_token
                        )
                        per_iteration_tokens_per_second.append(model.tokens_per_second)
                        per_iteration_peak_wset.append(model.peak_wset)

                # Report progress
                report_progress_fn((iteration + 1) / total_iterations)

            except Exception as e:
                raise RuntimeError(f"Benchmark iteration failed: {str(e)}") from e

        # Validate we have enough data
        if not per_iteration_time_to_first_token or not per_iteration_tokens_per_second:
            raise Bench.not_enough_tokens(output_tokens)

        # Record input token count (from the last successful iteration)
        if model.prompt_tokens is not None:
            self.input_ids_len_list.append(model.prompt_tokens)
        else:
            # Estimate if server didn't provide it
            # This is a rough estimate based on typical tokenization
            estimated_tokens = len(prompt.split()) + 1
            self.input_ids_len_list.append(estimated_tokens)

        # Calculate and store statistics
        mean_time_to_first_token = statistics.mean(per_iteration_time_to_first_token)
        self.mean_time_to_first_token_list.append(mean_time_to_first_token)

        # Prefill tokens per second
        input_tokens = self.input_ids_len_list[-1]
        self.prefill_tokens_per_second_list.append(
            input_tokens / mean_time_to_first_token
        )

        # Token generation speed
        self.token_generation_tokens_per_second_list.append(
            statistics.mean(per_iteration_tokens_per_second)
        )

        # Standard deviation for TTFT
        try:
            self.std_dev_time_to_first_token_list.append(
                statistics.stdev(per_iteration_time_to_first_token)
            )
        except StatisticsError:
            # Less than 2 measurements
            self.std_dev_time_to_first_token_list.append(None)

        # Standard deviation for TPS
        try:
            self.std_dev_token_generation_tokens_per_second_list.append(
                statistics.stdev(per_iteration_tokens_per_second)
            )
        except StatisticsError:
            # Less than 2 measurements
            self.std_dev_token_generation_tokens_per_second_list.append(None)

        # Calculate max memory used from wrapped server processes
        if self.save_max_memory_used:
            filtered_list = [
                item for item in per_iteration_peak_wset if item is not None
            ]
            if filtered_list:
                mean_gb_used = statistics.mean(filtered_list) / 1024**3
                self.max_memory_used_gb_list.append(mean_gb_used)
            else:
                self.max_memory_used_gb_list.append(None)


# This file was originally licensed under Apache 2.0. It has been modified.
# Modifications Copyright (c) 2025 AMD
