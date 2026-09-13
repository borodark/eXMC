"""Gate 1b: PyMC's JAX-compiled model against its C-compiled model.

  bench/pymc_race/.venv-jax/bin/python bench/pymc_race/parity_jax.py

The numpyro and blackjax arms sample the SAME PyMC model, but PyTensor compiles
it for JAX instead of C, and the two backends do not share arithmetic (lgamma,
log1p, softplus and the gradient graph are each backend's own kernels). This
checks that the density and gradient the JAX arms sample from are the ones the
other arms sample from, before any timing.

Points, per model, in PyMC's unconstrained space:
  - 200 around the initial point, N(0, 0.5) noise, gate 1's seed (parity.py);
  - the 200 posterior points gate 1 was redone at
    (reference/parity_posterior_points.json, from parity_posterior.py), mapped
    through each RV's own transform.

At each point: `model.compile_logp()` and `model.compile_dlogp()` (Jacobian
included, as the samplers use them), default mode against mode="JAX" with
`jax_enable_x64`. Residual: |c - j| / max(|c|, |j|, 1), worst over points and,
for the gradient, over components. PASS: both below 1e-9.
"""

import json
import os
import sys

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402
import pytensor  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from models import MODELS, build  # noqa: E402

HERE = os.path.dirname(__file__)
N_POINTS = 200
SEED = 20260913  # parity.py's
TOL = 1e-9


def residual(c, j):
    c, j = np.asarray(c, dtype=float), np.asarray(j, dtype=float)
    return float(np.max(np.abs(c - j) / np.maximum(np.maximum(np.abs(c), np.abs(j)), 1.0)))


def main():
    data = json.load(open(os.path.join(HERE, "data.json")))
    post_path = os.path.join(HERE, "reference", "parity_posterior_points.json")
    posterior = json.load(open(post_path))["models"] if os.path.exists(post_path) else {}
    ok_all = True
    rows = []

    for name in MODELS:
        model, _ = build(name, data)
        rvs = model.free_RVs
        values = [model.rvs_to_values[rv] for rv in rvs]

        logp_c, dlogp_c = model.compile_logp(), model.compile_dlogp()
        logp_j, dlogp_j = model.compile_logp(mode="JAX"), model.compile_dlogp(mode="JAX")

        rng = np.random.default_rng(SEED)
        ip = model.initial_point()
        points = [{v.name: ip[v.name] + rng.normal(0.0, 0.5, size=np.shape(ip[v.name])) for v in values}
                  for _ in range(N_POINTS)]

        n_post = 0
        if name in posterior:
            cons = [pytensor.tensor.dvector(f"c_{i}") for i, _ in enumerate(rvs)]
            forwards = []
            for rv, c in zip(rvs, cons):
                tr = model.rvs_to_transforms.get(rv)
                shaped = c if rv.ndim > 0 else c[0]
                forwards.append(shaped if tr is None else tr.forward(shaped, *rv.owner.inputs))
            to_unconstrained = pytensor.function(cons, forwards, on_unused_input="ignore")
            for p in posterior[name]["points"]:
                u = to_unconstrained(*[np.asarray(p[rv.name], dtype=float) for rv in rvs])
                points.append({v.name: np.asarray(val, dtype=float) for v, val in zip(values, u)})
                n_post += 1

        worst_logp = max(residual(logp_c(p), logp_j(p)) for p in points)
        worst_grad = max(residual(dlogp_c(p), dlogp_j(p)) for p in points)
        verdict = "PASS" if worst_logp < TOL and worst_grad < TOL else "FAIL"
        ok_all &= verdict == "PASS"
        rows.append((name, len(points) - n_post, n_post, worst_logp, worst_grad, verdict))
        print(f"{name:14s} points {len(points) - n_post}+{n_post}  logp {worst_logp:.1e}  dlogp {worst_grad:.1e}  {verdict}",
              flush=True)

    print("\n| model | init points | posterior points | worst logp residual | worst dlogp residual | result |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]:.1e} | {r[4]:.1e} | {r[5]} |")
    print(f"\njax {jax.__version__}, x64 {jax.config.jax_enable_x64}, devices {jax.devices()}")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
