# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the MoEKernelOracle ABC introduced in PR series for #37753.

This file contains a single canonical demonstration that
`UnquantizedMoEKernelOracle` methods delegate one-to-one to the
existing module-level functions in `oracle/unquantized.py`. Each method
on `UnquantizedMoEKernelOracle` follows the same `return module_fn(args)`
pattern, so verifying delegation for one method (`make_kernel`) gives
high confidence in the rest.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
from vllm.model_executor.layers.fused_moe.oracle import UnquantizedMoEKernelOracle
from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
    UnquantizedMoeBackend,
)


@pytest.mark.parametrize(
    "da_enabled,parallel_sizes,expect_modular",
    [
        ("1", (1, 1, 1), True),
        ("0", (1, 1, 1), False),
        (None, (1, 1, 1), False),
        ("1", (1, 2, 2), False),
    ],
)
def test_nvfp4_da_uses_precomputed_routing_on_single_rank(
    monkeypatch, da_enabled, parallel_sizes, expect_modular
):
    """DA must select modular W4A4 only on a single rank."""
    from vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe import (
        TrtLlmNvFp4ExpertsModular,
        TrtLlmNvFp4ExpertsMonolithic,
    )
    from vllm.model_executor.layers.fused_moe.oracle import nvfp4
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kNvfp4Dynamic,
        kNvfp4Static,
    )

    if da_enabled is None:
        monkeypatch.delenv("FLASHINFER_DIST_AWARE_AUTOTUNE", raising=False)
    else:
        monkeypatch.setenv("FLASHINFER_DIST_AWARE_AUTOTUNE", da_enabled)
    tp_size, dp_size, ep_size = parallel_sizes
    config = SimpleNamespace(
        moe_backend="flashinfer_trtllm",
        swiglu_limit=None,
        moe_parallel_config=SimpleNamespace(
            tp_size=tp_size,
            dp_size=dp_size,
            ep_size=ep_size,
            use_batched_activation_format=False,
        ),
    )
    for cls in (TrtLlmNvFp4ExpertsModular, TrtLlmNvFp4ExpertsMonolithic):
        monkeypatch.setattr(
            cls, "is_supported_config", staticmethod(lambda *args: (True, None))
        )
    backend, selected = nvfp4.select_nvfp4_moe_backend(
        config, kNvfp4Static, kNvfp4Dynamic
    )
    assert backend == nvfp4.NvFp4MoeBackend.FLASHINFER_TRTLLM
    assert selected is (
        TrtLlmNvFp4ExpertsModular if expect_modular else TrtLlmNvFp4ExpertsMonolithic
    )


def test_trtllm_nvfp4_does_not_admit_weight_only_quantization():
    """Changing dispatch must not reinterpret a W4A16 checkpoint as W4A4."""
    from vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe import (
        TrtLlmNvFp4ExpertsModular,
        TrtLlmNvFp4ExpertsMonolithic,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import kNvfp4Static

    for cls in (TrtLlmNvFp4ExpertsModular, TrtLlmNvFp4ExpertsMonolithic):
        assert not cls._supports_quant_scheme(kNvfp4Static, None)


class TestUnquantizedDelegation:
    """UnquantizedMoEKernelOracle methods must delegate to the existing
    module-level functions; behaviour is bit-identical."""

    def test_make_kernel_delegates(self) -> None:
        quant_config = object()
        moe_config = object()
        experts_cls = TritonExperts
        sentinel_kernel = object()

        with patch(
            "vllm.model_executor.layers.fused_moe.oracle.unquantized."
            "make_unquantized_moe_kernel",
            return_value=sentinel_kernel,
        ) as mocked:
            out = UnquantizedMoEKernelOracle().make_kernel(
                quant_config,
                moe_config,
                UnquantizedMoeBackend.TRITON,
                experts_cls,
            )

        mocked.assert_called_once_with(
            quant_config,
            moe_config,
            UnquantizedMoeBackend.TRITON,
            experts_cls,
            None,  # routing_tables default
        )
        assert out is sentinel_kernel
