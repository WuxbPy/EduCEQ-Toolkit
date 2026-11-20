# -*- coding: utf-8 -*-
"""Cost of the pipeline against object count, and against two baselines (Section 5.9)

The question is what the pipeline costs in time and memory as the data grow, and
how that cost compares with existing practice. The released code depends only on
numpy and scipy, so the comparison that is meaningful is with the two things a
reader would otherwise write: a dense pairwise implementation, and the published
Leslie-Kronenfeld formula on a nearest-neighbour list, which is what sfdep
implements. A direct call against sfdep would compare a Python sparse kernel with
an R neighbour list on different hardware, which is why the parity check of
Section 5.5 is a definitional one and this section is a cost one.

Every dataset is drawn at a fixed density of 100 objects per square kilometre, so
that the number of neighbours per object is constant and the cost is expected to
grow linearly in N; the bandwidth is 0.12 km in every row.

The benchmark relaxes both bounds of the check (ess_min = 1, rho_max = 0.2) so
that the estimator is reached at every N and the same code path is timed in every
row; the decision itself is not what is measured here, and no number in this file
should be read as a verdict.

Run:
    python evaluation/benchmarks.py
Output: results/benchmarks.csv and results/benchmarks.json
"""
import json
import os
import sys
import threading
import time

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from educeq import kernels, diagnostics, estimator, inference  # noqa: E402

RES = os.path.join(os.path.dirname(os.path.dirname(HERE)), "results")
SIZES = [1000, 3000, 10000, 30000, 100000]
DENSITY = 100.0
BAND = 0.12
N_PERM = 99
SEED = 20261114


class Peak:
    """Peak resident set size in MB while the block runs."""

    def __init__(self):
        self.peak = 0.0
        self._run = True

    def __enter__(self):
        import psutil
        p = psutil.Process()
        base = p.memory_info().rss
        self.peak = 0.0

        def watch():
            while self._run:
                self.peak = max(self.peak, p.memory_info().rss - base)
                time.sleep(0.005)

        self.th = threading.Thread(target=watch, daemon=True)
        self.th.start()
        return self

    def __exit__(self, *a):
        self._run = False
        self.th.join()
        self.peak /= 1e6


def timeit(fn, repeat=3):
    best = np.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best, out


def bench_toolkit(n, rng):
    side = np.sqrt(n / DENSITY)
    xy = rng.uniform(0, side, size=(n, 2))
    types = rng.integers(0, 3, n)
    row = dict(N=n)
    with Peak() as pk:
        tb, W = timeit(lambda: kernels.distance(xy, bandwidth=BAND))
        tc, dec = timeit(lambda: diagnostics.check(W, types, ess_min=1.0,
                                                   rho_max=0.2))
        te, C = timeit(lambda: estimator.ceq_matrix(W, types, contract=dec))
        tp, _ = timeit(lambda: inference.permutation(W, types, contract=dec,
                                                     n_perm=N_PERM, seed=SEED),
                       repeat=1)
        row.update(nnz=int(W.nnz), nnz_per_object=round(W.nnz / n, 1),
                   build=tb, check=tc, estimate=te, perm99=tp, peak_MB=pk.peak,
                   decidable=bool(dec.decidable))
    print("N=%6d  nnz=%9d  build=%6.2fs  check=%6.2fs  ceq=%5.2fs  "
          "perm99=%6.2fs  peak=%7.1f MB" % (n, row["nnz"], tb, tc, te, tp, pk.peak),
          flush=True)
    return row


def bench_dense(n, rng):
    """Dense pairwise kernel and quotient, the implementation a reader writes first."""
    side = np.sqrt(n / DENSITY)
    xy = rng.uniform(0, side, size=(n, 2))
    types = rng.integers(0, 3, n)
    K = 3
    p = np.bincount(types, minlength=K) / n

    def run():
        d2 = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)
        W = np.exp(-d2 / (2 * BAND ** 2))
        W[d2 > (3 * BAND) ** 2] = 0.0
        np.fill_diagonal(W, 0.0)
        onehot = np.zeros((n, K))
        onehot[np.arange(n), types] = 1.0
        contrib = W @ onehot
        tot = W.sum(1)
        C = np.zeros((K, K))
        for a in range(K):
            idx = types == a
            C[a] = contrib[idx].sum(0) / (tot[idx].sum() * p)
        return C

    with Peak() as pk:
        t, _ = timeit(run, repeat=1)
    print("dense  N=%6d  %7.2fs  peak=%7.1f MB" % (n, t, pk.peak), flush=True)
    return dict(N=n, seconds=t, peak_MB=pk.peak)


def bench_knn(n, k=8, rng=None):
    """Leslie-Kronenfeld quotient on a k nearest-neighbour list, as sfdep has it."""
    side = np.sqrt(n / DENSITY)
    xy = rng.uniform(0, side, size=(n, 2))
    types = rng.integers(0, 3, n)
    K = 3
    p = np.bincount(types, minlength=K) / n

    def run():
        tree = cKDTree(xy)
        _, idx = tree.query(xy, k=k + 1)
        idx = idx[:, 1:]
        same = (types[idx] == types[:, None])
        C = np.zeros((K, K))
        for a in range(K):
            ia = np.where(types == a)[0]
            nb = idx[ia]
            tot = same[ia].sum()
            for b in range(K):
                C[a, b] = (types[nb] == b).sum() / (tot * p[b])
        return C

    with Peak() as pk:
        t, _ = timeit(run, repeat=1)
    print("knn    N=%6d  %7.2fs  peak=%7.1f MB" % (n, t, pk.peak), flush=True)
    return dict(N=n, seconds=t, peak_MB=pk.peak)


def main():
    os.makedirs(RES, exist_ok=True)
    rows = [bench_toolkit(n, np.random.default_rng(SEED + n)) for n in SIZES]
    dense = [bench_dense(n, np.random.default_rng(SEED + n)) for n in (1000, 2000, 4000)]
    knn = [bench_knn(n, rng=np.random.default_rng(SEED + n)) for n in (1000, 10000)]
    import csv
    with open(os.path.join(RES, "benchmarks.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    out = dict(toolkit=rows, dense=dense, knn=knn, bandwidth=BAND, density=DENSITY,
               n_perm=N_PERM)
    with open(os.path.join(RES, "benchmarks.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("dense cost per object grows as N: %.1fx from 1e3 to 4e3"
          % (dense[2]["seconds"] / dense[0]["seconds"]), flush=True)


if __name__ == "__main__":
    main()
