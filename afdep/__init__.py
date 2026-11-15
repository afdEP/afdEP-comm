# SPDX-License-Identifier: Apache-2.0
"""Expert-parallel communication for Attention-FFN Disaggregation."""

from .deepep import Buffer, Config, DispatchHandle, Lanes, ReceiveResult

__version__ = "1.1.0"
__all__ = [
    "Buffer",
    "Config",
    "DispatchHandle",
    "Lanes",
    "ReceiveResult",
    "__version__",
]
