# Bayesian Statistical Process Control with eXMC

## The State of Affairs

Statistical Process Control has been frequentist since Walter Shewhart drew his
first control chart at Bell Labs in 1924. A century later, the tools have barely
changed: plot the sample mean, draw lines at three sigma, investigate when a
point crosses. The Shewhart chart, the CUSUM, the EWMA — all assume the process
parameters are known and fixed, estimated from a Phase I dataset of 80-150
observations that somebody collected before production began.

This assumption was reasonable when Western Electric manufactured rotary phones
in lots of ten thousand. It is less reasonable when a semiconductor fab runs
fifty wafers of a custom design, when a pharmaceutical company produces a
biologic in batches of twelve, or when an IoT sensor streams data from equipment
that has never been calibrated against a reference.

Bayesian methods address this by treating process parameters as uncertain
quantities with distributions, not fixed numbers with confidence intervals. The
posterior updates with each new observation. The predictive distribution for the
next measurement naturally widens when little data is available and tightens as
evidence accumulates. Prior knowledge from similar processes, vendor
specifications, or engineering judgment enters the model as an informative prior
rather than being discarded in favor of "letting the data speak."

The literature has been building toward this for thirty years, but adoption
remains glacial. The reasons are not statistical — they are computational and
cultural.

### Key Ideas

**Conjugate Sequential Monitoring** (no MCMC needed):

The Normal-Normal model for a process mean with known variance updates in closed
form. Given prior N(mu_0, tau_0^2) and observation x with known variance
sigma^2, the posterior is:

    mu_1 = (mu_0/tau_0^2 + x/sigma^2) / (1/tau_0^2 + 1/sigma^2)
    tau_1^2 = 1 / (1/tau_0^2 + 1/sigma^2)

Each observation costs one division, two additions, and one reciprocal. The
predictive distribution for the next observation is N(mu_n, tau_n^2 + sigma^2)
— a Bayesian control chart that self-calibrates.

For unknown variance, the Normal-Inverse-Gamma conjugate gives a Student-t
predictive with heavier tails that automatically account for parameter
uncertainty. This matters most in the first 20-30 observations, exactly where
traditional charts are most unreliable.

**Online Changepoint Detection** (Adams & MacKay, 2007):

The BOCPD algorithm maintains a posterior distribution over "run length" — the
number of observations since the last changepoint. At each time step, it
computes the predictive likelihood under each possible run length and updates
via a message-passing scheme. The algorithm is O(1) per observation with
truncation and supports any exponential family observation model.

This is the natural Bayesian replacement for the CUSUM chart: instead of
accumulating evidence for a shift in a frequentist statistic, it computes the
posterior probability that a shift has occurred.

**Full MCMC for Complex Models**:

When the process model is non-conjugate (e.g., multiple changepoints with
unknown number, hierarchical multi-stream monitoring, non-normal data with
complex correlation structure), NUTS sampling provides the general solution.
This is where eXMC's existing infrastructure applies directly.

### The Software Gap

| Language | Packages | Limitation |
|----------|----------|-----------|
| R | `bcp`, `ocp`, `bayespm`, `bayesWatch`, `mcp` | Single-threaded, batch-oriented, no streaming |
| Python | `bayesian-changepoint-detection`, PyMC, Stan | Same — designed for offline analysis |
| Stan | Change-point models in user guide | Compilation latency, no online updating |
| **Elixir/BEAM** | **Nothing** | The gap eXMC fills |

The BEAM's concurrency model is uniquely suited for production SPC: one
GenServer per monitored stream, conjugate updates in microseconds, fault-tolerant
supervision, and the ability to monitor thousands of streams simultaneously on a
single node. No existing SPC software can do this.

### Classic Datasets

**Nile River Annual Flow (1871-1970)**:
100 annual measurements of flow at Aswan. Known changepoint near 1898 (dam
construction). The canonical benchmark for changepoint detection. Available in R
as `datasets::Nile` and Python as `statsmodels.datasets.nile`.

**Montgomery's Piston Ring Data**:
40 samples of 5, measuring inside diameter. Samples 1-25 are in-control (Phase
I), 26-40 are Phase II. From Douglas Montgomery's *Introduction to Statistical
Quality Control*. Available in R as `qcc::pistonrings`.

**UK Coal Mining Disasters (1851-1962)**:
Annual counts of disasters with 10+ deaths. Poisson model with unknown
switchpoint. The PyMC tutorial dataset.

### References

- Adams, R. P. & MacKay, D. J. C. (2007). "Bayesian Online Changepoint
  Detection." arXiv:0710.3742.
- Barry, D. & Hartigan, J. A. (1993). "A Bayesian Analysis for Change Point
  Problems." *JASA*, 88, 309-319.
- Bourazas, K., Kiagias, D. & Tsiamyrtzis, P. (2022). "Predictive Control
  Charts (PCC): A Bayesian Approach in Online Monitoring of Short Runs."
  *Journal of Quality Technology*, 54(4).
- Fearnhead, P. (2006). "Exact and Efficient Bayesian Inference for Multiple
  Changepoint Problems." *Statistics and Computing*, 16, 203-213.
- Apley, D. W. (2012). "Posterior Distribution Charts: A Bayesian Approach for
  Graphically Exploring a Process Mean." *Technometrics*, 54(3), 279-310.
- Menzefricke, U. (2002). "On the Evaluation of Control Chart Limits Based on
  Predictive Distributions." *Comm. in Statistics*, 31(8), 1423-1440.
- Montgomery, D. C. (2019). *Introduction to Statistical Quality Control*,
  8th ed. Wiley.
- Tsiamyrtzis, P. (2007). "Statistical Process Control, Bayesian."
  *Encyclopedia of Statistics in Quality and Reliability*, Wiley.
