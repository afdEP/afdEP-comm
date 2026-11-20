# SPDX-License-Identifier: Apache-2.0
"""Expert-parallel communication for Attention-FFN Disaggregation."""

from .deepep import Buffer, Config, DispatchHandle, Lanes, ReceiveResult

__version__ = "1.2.1"
__all__ = [
    "Buffer",
    "Config",
    "DispatchHandle",
    "Lanes",
    "ReceiveResult",
    "__version__",
]
