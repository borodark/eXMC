"""Gate 1 at the POSTERIOR: log-density parity at points drawn from the reference runs.

  bench/pymc_race/.venv/bin/python bench/pymc_race/parity_posterior.py
  PARITY_POINTS=bench/pymc_race/reference/parity_posterior_points.json \
    MIX_ENV=test mix run --no-deps-check --no-compile bench/pymc_race/parity.exs

parity.py draws its points around PyMC's initial point plus noise. For SV
that meant sigma of about 0.007-0.05 and s near 0, while the posterior sits
at sigma about 0.08 and s about 1. A density difference confined to where the
posterior actually lives would pass that gate. Found 2026-09-13, when the SV
references disagreed: exmc's sigma was 7% higher and s's sd 9% wider than
PyMC's, consistently across chains. The density agreed at the posterior
(about 9e-6 nats). The defect was in the sampler: the speculative subtree NIF
built backward subtrees with swapped endpoints. It was fixed in 509e22b26, and
the SV references now agree.

Here the points are 100 draws from EACH framework's reference posterior per
model (reference/pymc_<model>.npz, reference/exmc_<model>__<var>.bin),
seeded. For each, the constrained value is mapped into PyMC's unconstrained
space through the RV's own transform, and PyMC's log density WITHOUT the
Jacobian is recorded. parity.exs consumes the same file format unchanged.
"""

import json
import os
import sys

import numpy as np
import pytensor

sys.path.insert(0, os.path.dirname(__file__))
from models import MODELS, build  # noqa: E402
import reference_compare as rc  # noqa: E402

HERE = os.path.dirname(__file__)
PER_SOURCE = 100
SEED = 20260914


def draws_at(draws, idx):
    """{var: (chains, draws, ...)} -> list of {var: flat list} at (chain, draw) indices."""
    return [{v: np.atleast_1d(a[c, d]).astype(float).tolist() for v, a in draws.items()} for c, d in idx]


def main():
    data = json.load(open(os.path.join(HERE, "data.json")))
    rng = np.random.default_rng(SEED)
    out = {"n_points": 2 * PER_SOURCE, "seed": SEED, "source": "reference posteriors", "models": {}}

    for name in MODELS:
        pm_draws = rc.load_pymc(name)
        ex_draws, _meta = rc.load_exmc(name)
        model, var_names = build(name, data)

        points = []
        for src in (pm_draws, ex_draws):
            first = next(iter(src.values()))
            idx = list(zip(rng.integers(0, first.shape[0], PER_SOURCE), rng.integers(0, first.shape[1], PER_SOURCE)))
            points += draws_at(src, idx)

        rvs = model.free_RVs
        values = [model.rvs_to_values[rv] for rv in rvs]
        logp_nojac = model.compile_logp(jacobian=False)

        # constrained -> unconstrained, through each RV's own transform
        cons = [pytensor.tensor.dvector(f"c_{i}") for i, _ in enumerate(rvs)]
        forwards = []
        for rv, c in zip(rvs, cons):
            tr = model.rvs_to_transforms.get(rv)
            shaped = c if rv.ndim > 0 else c[0]
            forwards.append(shaped if tr is None else tr.forward(shaped, *rv.owner.inputs))
        to_unconstrained = pytensor.function(cons, forwards, on_unused_input="ignore")

        logps = []
        for p in points:
            u = to_unconstrained(*[np.asarray(p[rv.name], dtype=float) for rv in rvs])
            logps.append(float(logp_nojac({v.name: val for v, val in zip(values, u)})))

        out["models"][name] = {"var_names": var_names, "points": points, "logp_nojac": logps}
        print(f"{name:14s} {len(points)} posterior points, logp range [{min(logps):.3f}, {max(logps):.3f}]")

    path = os.path.join(HERE, "reference", "parity_posterior_points.json")
    with open(path, "w") as f:
        json.dump(out, f)
    print("wrote", path)


if __name__ == "__main__":
    main()
