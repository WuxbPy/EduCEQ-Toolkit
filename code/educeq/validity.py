# -*- coding: utf-8 -*-
"""educeq.validity — whether a conclusion may be used (module five)

This is the validity module of the paper. The measure answers "what is the
structure"; this module answers "may that conclusion be used". Keeping the two
apart is itself a design decision.

Three questions:

  Q1 does the association exist          partial_corr / mde_corr / tost_corr
  Q2 is it the association of interest   proxy_test / mediation_boot /
                                         classify_proxy
  Q3 can it discriminate individuals     discrimination_decomposition /
                                         dd_bootstrap

The equivalence test (TOST) behind Q1 is the tool used most often in the paper:
it replaces "are the two significantly different" with "can we assert the
difference is below a threshold of practical relevance", which separates
insufficient evidence from evidence against.
"""
from ._infer import (partial_corr, mde_corr, tost_corr, partial_r2,
                     proxy_test, classify_proxy, mediation_boot,
                     discrimination_decomposition, dd_bootstrap, infer_report)

__all__ = ["partial_corr", "mde_corr", "tost_corr", "partial_r2",
           "proxy_test", "classify_proxy", "mediation_boot",
           "discrimination_decomposition", "dd_bootstrap", "infer_report"]
