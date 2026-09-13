"""Step 4, the check: do the PyMC and eXMC reference posteriors agree?

  bench/pymc_race/.venv/bin/python bench/pymc_race/reference_compare.py

Reads reference/pymc_<model>.npz and reference/exmc_<model>__<var>.bin (+ .json),
and computes, with ArviZ for BOTH frameworks, per scalar component: mean, sd,
MCSE of the mean and of the sd, bulk ESS and rank-normalised R-hat.

Agreement, per component:
  z_mean = |mean_p - mean_e| / sqrt(mcse_mean_p^2 + mcse_mean_e^2)
  z_sd   = |sd_p   - sd_e  | / sqrt(mcse_sd_p^2   + mcse_sd_e^2)
A model AGREES when max z_mean and max z_sd are below Z_MAX (5; with up to 102
components per model, a few z near 3 are ordinary), and both frameworks' R-hat
are below 1.01 and bulk ESS above 1000 for every component.

The funnel is also checked against its analytic marginals: y ~ N(0, 3), and
x_i with mean 0 and variance E[exp(y)] = exp(4.5). The x variance has heavy
tails, so its z is reported but not gated.

Writes reference/summary.json: for each agreeing model, the POOLED reference
(mean and sd over both frameworks' draws) with a combined MCSE. This is what
gate 2 scores race runs against. A model that disagrees gets no reference, and
the race waits for the disagreement to be explained.
"""

import json
import os
import sys

import arviz as az
import numpy as np

HERE = os.path.dirname(__file__)
REF = os.path.join(HERE, "reference")
# A second host's run, copied back: EXMC_REF / PYMC_REF name the directories
# holding each framework's draws; summary.json goes into EXMC_REF.
EXMC_REF = os.environ.get("EXMC_REF", REF)
PYMC_REF = os.environ.get("PYMC_REF", REF)
MODELS = ["simple", "medium", "stress", "eight_schools", "funnel", "logistic", "sv"]
Z_MAX = 5.0


def load_exmc(name):
    meta = json.load(open(os.path.join(EXMC_REF, f"exmc_{name}.json")))
    out = {}
    for var, shape in meta["shapes"].items():
        raw = np.fromfile(os.path.join(EXMC_REF, f"exmc_{name}__{var}.bin"), dtype="<f8")
        arr = raw.reshape(shape)  # (chains, draws, k)
        out[var] = arr[..., 0] if shape[-1] == 1 else arr
    return out, meta


def load_pymc(name):
    z = np.load(os.path.join(PYMC_REF, f"pymc_{name}.npz"))
    return {k: z[k] for k in z.files}


def stats(draws):
    dt = az.from_dict({"posterior": draws})
    ess = az.ess(dt, method="bulk")
    rhat = az.rhat(dt)
    m_mean = az.mcse(dt, method="mean")
    m_sd = az.mcse(dt, method="sd")
    out = {}
    for var, arr in draws.items():
        flat = arr.reshape(arr.shape[0] * arr.shape[1], -1)
        out[var] = {
            "mean": flat.mean(axis=0),
            "sd": flat.std(axis=0, ddof=1),
            "ess_bulk": np.atleast_1d(np.asarray(ess[var].values)),
            "rhat": np.atleast_1d(np.asarray(rhat[var].values)),
            "mcse_mean": np.atleast_1d(np.asarray(m_mean[var].values)),
            "mcse_sd": np.atleast_1d(np.asarray(m_sd[var].values)),
        }
    return out


def main(names):
    summary = {"z_max": Z_MAX, "models": {}}
    ok_all = True
    print("| model | components | max z mean | max z sd | worst R-hat (pymc / exmc) | min ESS (pymc / exmc) | divergences exmc | verdict |")
    print("|---|---|---|---|---|---|---|---|")

    for name in names:
        try:
            pm_draws = load_pymc(name)
            ex_draws, ex_meta = load_exmc(name)
        except FileNotFoundError as e:
            print(f"| {name} | | | | | | | MISSING ({os.path.basename(e.filename)}) |")
            ok_all = False
            continue

        missing = sorted(set(pm_draws) ^ set(ex_draws))
        if missing:
            print(f"| {name} | | | | | | | VARIABLE MISMATCH {missing} |")
            ok_all = False
            continue

        sp, se = stats(pm_draws), stats(ex_draws)
        zm, zs, rh_p, rh_e, ess_p, ess_e, n = [], [], [], [], [], [], 0
        ref = {}
        for var in sorted(sp):
            a, b = sp[var], se[var]
            zm.append(np.abs(a["mean"] - b["mean"]) / np.sqrt(a["mcse_mean"] ** 2 + b["mcse_mean"] ** 2))
            zs.append(np.abs(a["sd"] - b["sd"]) / np.sqrt(a["mcse_sd"] ** 2 + b["mcse_sd"] ** 2))
            rh_p.append(a["rhat"]); rh_e.append(b["rhat"]); ess_p.append(a["ess_bulk"]); ess_e.append(b["ess_bulk"])
            n += a["mean"].size
            pooled = np.concatenate([
                pm_draws[var].reshape(-1, *pm_draws[var].shape[2:]),
                ex_draws[var].reshape(-1, *ex_draws[var].shape[2:]),
            ])
            ref[var] = {
                "mean": np.atleast_1d(pooled.mean(axis=0)).tolist(),
                "sd": np.atleast_1d(pooled.std(axis=0, ddof=1)).tolist(),
                "mcse_mean": (np.sqrt(a["mcse_mean"] ** 2 + b["mcse_mean"] ** 2) / 2).tolist(),
            }

        cat = lambda xs: np.concatenate([np.atleast_1d(x) for x in xs])
        max_zm, max_zs = float(cat(zm).max()), float(cat(zs).max())
        worst_rp, worst_re = float(cat(rh_p).max()), float(cat(rh_e).max())
        min_ep, min_ee = float(cat(ess_p).min()), float(cat(ess_e).min())

        agrees = max_zm < Z_MAX and max_zs < Z_MAX
        converged = worst_rp < 1.01 and worst_re < 1.01 and min_ep > 1000 and min_ee > 1000
        verdict = "AGREE" if agrees and converged else ("DISAGREE" if not agrees else "NOT CONVERGED")

        extra = {}
        if name == "funnel":
            y_ref, x_var = (0.0, 3.0), float(np.exp(4.5))
            for label, st in (("pymc", sp), ("exmc", se)):
                y = st["y"]
                extra[label] = {
                    "y_mean_z": float(abs(y["mean"][0] - y_ref[0]) / y["mcse_mean"][0]),
                    "y_sd": float(y["sd"][0]),
                    "x_var_mean_over_truth": float(np.mean(st["x"]["sd"] ** 2) / x_var),
                }
            if max(extra["pymc"]["y_mean_z"], extra["exmc"]["y_mean_z"]) > Z_MAX:
                verdict = "OFF ANALYTIC"

        print(f"| {name} | {n} | {max_zm:.2f} | {max_zs:.2f} | {worst_rp:.4f} / {worst_re:.4f} | "
              f"{min_ep:.0f} / {min_ee:.0f} | {ex_meta['divergences']} | {verdict} |")

        entry = {"verdict": verdict, "components": n, "max_z_mean": max_zm, "max_z_sd": max_zs,
                 "rhat_worst": {"pymc": worst_rp, "exmc": worst_re},
                 "ess_bulk_min": {"pymc": min_ep, "exmc": min_ee},
                 "exmc_wall_s": ex_meta["wall_s"], "exmc_divergences": ex_meta["divergences"],
                 "exmc_provenance": ex_meta.get("provenance")}
        if extra:
            entry["funnel_analytic"] = extra
        if verdict == "AGREE":
            entry["reference"] = ref
        else:
            ok_all = False
        summary["models"][name] = entry

    with open(os.path.join(EXMC_REF, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    if "funnel" in summary["models"] and "funnel_analytic" in summary["models"]["funnel"]:
        print("\nfunnel vs analytic:", json.dumps(summary["models"]["funnel"]["funnel_analytic"]))
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main(sys.argv[1:] or MODELS)
