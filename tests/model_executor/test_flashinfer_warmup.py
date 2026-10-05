# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.model_executor.warmup.kernel_warmup import _flashinfer_autotune_token_counts


@pytest.mark.parametrize("enabled,expected", [("0", (8192,)), ("1", (8192, 256))])
def test_da_autotune_covers_decode_graph_buckets(monkeypatch, enabled, expected):
    """Decode graphs need their own domain, not only the prefill shape."""
    monkeypatch.setenv("FLASHINFER_DIST_AWARE_AUTOTUNE", enabled)
    runner = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8192),
        vllm_config=SimpleNamespace(
            kernel_config=SimpleNamespace(
                moe_backend="flashinfer_trtllm", linear_backend="auto"
            ),
            compilation_config=SimpleNamespace(cudagraph_capture_sizes=[256, 8192]),
        ),
    )
    assert _flashinfer_autotune_token_counts(runner) == expected


def test_cutedsl_linear_warmup_remains_available_without_da(monkeypatch):
    """The MoE addition must preserve the small BF16 linear warmup."""
    monkeypatch.delenv("FLASHINFER_DIST_AWARE_AUTOTUNE", raising=False)
    runner = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=2048),
        vllm_config=SimpleNamespace(
            kernel_config=SimpleNamespace(
                moe_backend="auto", linear_backend="flashinfer_cutedsl"
            )
        ),
    )
    assert _flashinfer_autotune_token_counts(runner) == (2048, 32)
