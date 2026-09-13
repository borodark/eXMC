"""Gate 1, PyMC half: log densities at shared points, for parity.exs to compare.

  bench/pymc_race/.venv/bin/python bench/pymc_race/parity.py

For each model: 200 points, PyMC's initial point in its unconstrained space
plus N(0, 0.5) noise (seeded), mapped to CONSTRAINED values through each free
RV's own transform. Written to parity_points.json with PyMC's log density at
each point WITHOUT the Jacobian (`compile_logp(jacobian=False)`).

Constrained, not unconstrained, because the frameworks do not share a
parameter space: exmc samples HalfNormal through softplus, PyMC through log.
The joint density over the constrained values is the model; the Jacobian is
each framework's own business, and exmc's is subtracted on its side.
"""

import json
import os
import sys

import numpy as np
import pytensor

sys.path.insert(0, os.path.dirname(__file__))
if os.environ.get("PARITY_MODELS"):  # negative controls: an alternative models module
    sys.path.insert(0, os.environ["PARITY_MODELS"])
from models import MODELS, build  # noqa: E402

HERE = os.path.dirname(__file__)
N_POINTS = 200
SEED = 20260913


def main():
    data = json.load(open(os.path.join(HERE, "data.json")))
    out = {"n_points": N_POINTS, "seed": SEED, "models": {}}

    for name in MODELS:
        model, var_names = build(name, data)
        rng = np.random.default_rng(SEED)
        ip = model.initial_point()
        logp_nojac = model.compile_logp(jacobian=False)

        rvs = model.free_RVs
        values = [model.rvs_to_values[rv] for rv in rvs]
        backwards = []
        for rv, v in zip(rvs, values):
            tr = model.rvs_to_transforms.get(rv)
            backwards.append(v if tr is None else tr.backward(v, *rv.owner.inputs))
        to_constrained = pytensor.function(values, backwards, on_unused_input="ignore")

        points, logps = [], []
        for _ in range(N_POINTS):
            pt_u = {v.name: ip[v.name] + rng.normal(0.0, 0.5, size=np.shape(ip[v.name])) for v in values}
            constrained = to_constrained(*[pt_u[v.name] for v in values])
            points.append({rv.name: np.atleast_1d(np.asarray(c, dtype=float)).tolist()
                           for rv, c in zip(rvs, constrained)})
            logps.append(float(logp_nojac(pt_u)))

        out["models"][name] = {
            "var_names": var_names,
            "points": points,
            "logp_nojac": logps,
            "transforms": {rv.name: type(model.rvs_to_transforms.get(rv)).__name__ for rv in rvs},
        }
        print(f"{name:14s} {len(points)} points, logp range [{min(logps):.3f}, {max(logps):.3f}]")

    with open(os.environ.get("PARITY_POINTS", os.path.join(HERE, "parity_points.json")), "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()
