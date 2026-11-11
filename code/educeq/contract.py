# -*- coding: utf-8 -*-
"""educeq.contract — capability tokens for the runtime contract (Section 4.7).

A verdict is not advice: it is permission to compute. This module implements the
permission as a token. ``diagnostics.check`` issues a :class:`Decidability`
object carrying a fingerprint of the data the verdict was computed on, and every
function that can return an estimate requires such a token. Passing none raises
:class:`ContractRequired`; passing one issued for different data raises
:class:`ContractMismatch`; passing one whose verdict was negative raises
:class:`ContractRefused`, and the refusal carries the verdict with it.

Why a token rather than a boolean
---------------------------------
A boolean argument can be set by the caller on the strength of nothing. A token
can only be obtained by running the checker on that very data, so the sentence
"no estimate is returned without a passing verdict" becomes a property of the
API rather than a promise in the documentation. The property is enforced by
``tests/test_educeq.py::test_public_api_audit`` and ``test_no_core_bypass``,
which fail if a new public entry point drops the requirement.

The fingerprint is a digest of quantities that identify the data as seen by the
estimator — size, number of non-zero weights, the first two moments of the
weight distribution, the category counts — together with the thresholds and the
target category. It is not a cryptographic guarantee against a determined
caller, and it is not meant to be: it prevents the accidental reuse of a verdict
computed on one dataset for another, which is the realistic failure.
"""
import hashlib

import numpy as np

__all__ = ["ContractError", "ContractRequired", "ContractMismatch",
           "ContractRefused", "fingerprint", "require"]


class ContractError(RuntimeError):
    """Base class for violations of the runtime contract."""


class ContractRequired(ContractError, TypeError):
    """Raised when an entry point that can return an estimate is called without
    a verdict. Subclasses TypeError so that the usual ``except TypeError`` around
    API misuse keeps working."""


class ContractMismatch(ContractError, TypeError):
    """Raised when a verdict is presented for data other than those it was
    computed on."""


class ContractRefused(ContractError):
    """Raised when a verdict with ``decidable=False`` is presented.

    The verdict is available as ``exc.verdict``; it carries the failing steps,
    the diagnostic values and the advice string, so a caller that wants to log
    the reason does not have to re-run the checker.
    """

    def __init__(self, verdict):
        self.verdict = verdict
        super().__init__("the data are undecidable under the current criteria "
                         "(%s); no estimate is returned" % ", ".join(verdict.failed))


def fingerprint(W, types, **config):
    """Digest identifying the data a verdict was computed on.

    ``W`` is the weight matrix, ``types`` the category vector, and ``config`` the
    thresholds and the target category. Two calls with the same data and the same
    configuration produce the same digest; any change in size, sparsity, weight
    moments, category counts or configuration changes it.
    """
    coo = W.tocoo()
    h = hashlib.sha256()
    stats = np.asarray([W.shape[0], W.shape[1], coo.nnz,
                        float(coo.data.sum()),
                        float((coo.data ** 2).sum())], dtype=float)
    h.update(stats.tobytes())
    h.update(np.bincount(np.asarray(types, dtype=int)).tobytes())
    for k in sorted(config):
        h.update(("%s=%r;" % (k, config[k])).encode())
    return h.hexdigest()[:16]


def require(contract, W, types, **config):
    """Validate a verdict and return it, or raise.

    Parameters
    ----------
    contract : Decidability or None
        The object returned by :func:`educeq.diagnostics.check`.
    W, types :
        The data the estimate is about to be computed on.
    config :
        The thresholds and target recorded in the verdict.
    """
    if contract is None:
        raise ContractRequired(
            "this call can return an estimate and therefore requires a verdict: "
            "run dec = educeq.diagnostics.check(W, types, target=...) first and "
            "pass it as contract=dec")
    if not getattr(contract, "decidable", False):
        raise ContractRefused(contract)
    expected = getattr(contract, "fingerprint", None)
    if expected is not None:
        if not config:
            config = dict(target=getattr(contract, "target", None),
                          pair=getattr(contract, "pair", None),
                          ess_min=float(getattr(contract, "ess_min", 30.0)),
                          rho_max=float(getattr(contract, "rho_max", 0.01)))
        got = fingerprint(W, types, **config)
        if got != expected:
            raise ContractMismatch(
                "the verdict was issued for different data or a different "
                "configuration (token %s, data %s); re-run the checker on the "
                "data you are estimating" % (expected, got))
    return contract
