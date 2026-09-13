"""Step 4, PyMC half: long reference runs (docs/PYMC_RACE_PLAN.md, gate 2).

  bench/pymc_race/.venv/bin/python bench/pymc_race/reference_pymc.py [model ...]

4 chains x 10,000 draws after 2,000 tuning, target_accept 0.9, PyMC's own NUTS.
Draws go to reference/pymc_<model>.npz as (chains, draws, ...) arrays named by
the race's variable names; reference_compare.py checks them against exmc's.

Eight schools and the funnel are sampled NON-CENTERED here, because a reference
must be right before it is fast: the posterior over (mu, tau, theta) and over
(y, x) is identical, the geometry is not. theta and x are computed from the
draws (theta = mu + tau*z, x = exp(y/2)*z). posteriordb's eight schools
reference cannot be used: its prior is tau ~ HalfCauchy(5), this race's is
HalfNormal(5).
"""

import json
import os
import sys
import time

import numpy as np
import pymc as pm

sys.path.insert(0, os.path.dirname(__file__))
from models import build  # noqa: E402

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "reference")
KW = dict(draws=10_000, tune=2_000, chains=4, cores=4, target_accept=0.9,
          random_seed=2026, progressbar=False, compute_convergence_checks=False)

# PyMC 6 picks nutpie when it is installed, so the references are nutpie's
# unless REF_SAMPLER=pymc asks for PyMC's own NUTS. REF_TAG keeps the second
# opinion's files apart: reference/pymc<tag>_<model>.npz.
if os.environ.get("REF_SAMPLER"):
    KW["nuts_sampler"] = os.environ["REF_SAMPLER"]
TAG = os.environ.get("REF_TAG", "")


def eight_schools_nc(data):
    es = data["eight_schools"]
    y, sigma = np.array(es["y"], float), np.array(es["sigma"], float)
    with pm.Model() as m:
        mu = pm.Normal("mu", 0, 5)
        tau = pm.HalfNormal("tau", 5)
        z = pm.Normal("z", 0, 1, shape=8)
        pm.Normal("y_obs", mu + tau * z, sigma, observed=y)
    derive = lambda p: {"mu": p["mu"], "tau": p["tau"], "theta": p["mu"][..., None] + p["tau"][..., None] * p["z"]}
    return m, derive


def funnel_nc(data):
    with pm.Model() as m:
        pm.Normal("y", 0, 3)
        pm.Normal("z", 0, 1, shape=9)
    derive = lambda p: {"y": p["y"], "x": np.exp(p["y"][..., None] / 2) * p["z"]}
    return m, derive


def plain(name):
    def builder(data):
        m, var_names = build(name, data)
        return m, lambda p: {v: p[v] for v in var_names}
    return builder


BUILDERS = {
    "simple": plain("simple"),
    "medium": plain("medium"),
    "stress": plain("stress"),
    "eight_schools": eight_schools_nc,
    "funnel": funnel_nc,
    "logistic": plain("logistic"),
    "sv": plain("sv"),
}


def main(names):
    os.makedirs(OUT, exist_ok=True)
    data = json.load(open(os.path.join(HERE, "data.json")))
    for name in names:
        model, derive = BUILDERS[name](data)
        t0 = time.monotonic()
        with model:
            idata = pm.sample(**KW)
        wall = time.monotonic() - t0
        post = idata["posterior"]
        raw = {k: np.asarray(post[k].values) for k in post.data_vars}
        out = {k: np.asarray(v, dtype=float) for k, v in derive(raw).items()}
        div = int(np.asarray(idata["sample_stats"]["diverging"].values).sum())
        np.savez(os.path.join(OUT, f"pymc{TAG}_{name}.npz"), **out)
        print(f"{name:14s} wall {wall:7.1f} s  divergences {div}  vars {sorted(out)}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or list(BUILDERS))
