# posteriordb Validation

External validation of eXMC's NUTS sampler against [posteriordb](https://github.com/stan-dev/posteriordb) gold-standard reference posteriors (Stan draws).

## Results: 33/33 PASS (100%)

| Family | Models | Dimensions | Description |
|--------|--------|-----------|-------------|
| Eight Schools | 1 | d=10 | Hierarchical normal-normal (NCP) |
| Earnings | 7 | d=3-5 | Income regressions (log transforms, interactions) |
| KidIQ | 8 | d=3-5 | Parent-child cognitive development |
| Bayesian Linear Reg | 2 | d=6 | Correlated predictors, informative priors |
| Kilpisjarvi | 1 | d=3 | Temperature vs altitude |
| Mesquite | 6 | d=3-8 | Ecological biomass modeling |
| NES Elections | 8 | d=10 | US political ideology (1972-2000) |

## Pass Criteria (per parameter)

- **Mean**: Posterior mean within 0.5 standard deviations of reference mean
- **Variance**: Posterior SD within factor of 2 (0.5x-2.0x) of reference SD
- All parameters in a model must pass both criteria

This is the same validation methodology used by Stan, PyMC, and NumPyro.

## Running the Validation

```bash
cd exmc
mix deps.get

# CPU-only recommended for parallel chains
CUDA_VISIBLE_DEVICES="" mix run benchmark/posteriordb/validate_posteriordb.exs

# Use all available cores
CUDA_VISIBLE_DEVICES="" mix run benchmark/posteriordb/validate_posteriordb.exs --parallel 88

# Custom sample counts
CUDA_VISIBLE_DEVICES="" mix run benchmark/posteriordb/validate_posteriordb.exs --samples 2000 --warmup 2000
```

Default: 1000 warmup + 1000 sampling iterations per model, parallel across all CPU cores.

## Pre-computed Data

The `posteriordb_processed/` directory contains 33 pre-computed model specifications with:
- Response variable `y` and design matrix `X`
- Prior specifications
- 10,000 reference draws per parameter (10 Stan chains x 1,000 draws)

This allows running the validation without any Python dependencies.

### These are committed, and where they come from

The directory is tracked in git as of 2026-09-09. It used to be gitignored as
"regenerable", which held on a machine with python3, numpy and a clone of
posteriordb, and nowhere else. A fleet run on 2026-09-09 found the directory
absent on all three remote hosts: the harness listed zero posteriors and the
run reported success by having nothing to fail. `git pull` is now sufficient to
reproduce a verification run, which is the property the fleet needs.

**Provenance.** Every file here is derived from
[posteriordb](https://github.com/stan-dev/posteriordb) (stan-dev) by
`preprocess_posteriordb.py` — the response, design matrix, priors and reference
draws are theirs; the JSON shape is ours. Consult posteriordb's own licence and
citation guidance before redistributing this directory beyond this repository,
and note that this repository has a public remote.

~28 MB across 66 files, the largest being the `nes*` models at 1.4 MB each,
which is dominated by the 10,000 reference draws per parameter.

## Regenerating Preprocessed Data (optional)

If you want to regenerate the model specifications from the posteriordb source:

```bash
# Clone posteriordb
git clone --depth 1 https://github.com/stan-dev/posteriordb.git /tmp/posteriordb

# Preprocess (requires numpy)
pip install numpy
python benchmark/posteriordb/preprocess_posteriordb.py /tmp/posteriordb
```

## File Layout

```
benchmark/posteriordb/
  validate_posteriordb.exs       # Elixir validation script
  preprocess_posteriordb.py      # Python preprocessor (for regeneration)
  README.md                      # This file
  posteriordb_processed/
    manifest.json                # List of 33 posteriors
    validation_results.md        # Latest validation results
    *.json                       # 33 model specification files
```
