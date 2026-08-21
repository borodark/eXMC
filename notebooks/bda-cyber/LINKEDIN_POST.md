# LinkedIn Post — BDA for Cybersecurity

---

**The first probability textbook I owned was full of artillery problems.**

Shell dispersion. Range estimation. Target acquisition under fog. I loved it — I was nineteen and thought cannons were interesting. But probability theory is not about artillery. The examples kept most people out.

Every quantitative field has this bottleneck. The theory is universal — Bayes' theorem does not care whether you are estimating a sex ratio, a shell trajectory, or an IDS true positive rate. But the learner cares. A security analyst who sees "placenta previa" in Chapter 2 has to spend cognitive effort on a medical term before getting to the math. An epidemiologist who sees "artillery dispersion" has the same problem. The context is noise. The theory never arrives.

Here is the thing that has changed: **with models capable of good technical prose, there is no longer an excuse.**

If you can write a worked example about windshield hardness, you can write the same example about DNS query lengths. If you can write a decision theory problem about a jar of coins, you can write one about incident response at 2 AM — contain the server ($50K) or risk a missed breach ($2M) when your posterior says 15% probability of compromise. The math is identical. The motivation is entirely different.

I took all 8 chapters of Aki Vehtari's BDA Python demos — the canonical teaching material for Gelman et al.'s *Bayesian Data Analysis* — and rewrote every example for cybersecurity:

- **Beta-Binomial → IDS Rule Effectiveness.** "Your rule is 95% accurate. An alert fires. Probability it's real? Not 95%. It's 1.9%." Base rate neglect, explained with the math that every SOC analyst needs but no security certification teaches.
- **Normal model → DNS Baseline with DGA Contamination.** What does "normal" look like on your network? What happens when an adversary injects domain-generation-algorithm domains? The Gaussian breaks — visibly.
- **Eight Schools → Eight SOCs.** Branch offices report incident counts. HQ had 28. Southeast had 3. Is HQ worse, or do they just detect more? Is Southeast secure, or blind? Hierarchical partial pooling answers both.
- **Decision Analysis → Incident Response.** Contain, investigate, or ignore — computed from expected loss, not gut feeling. The optimal containment threshold is 2.5%, not 50%. The cost asymmetry demands it.
- **Posterior Predictive Checks → Threat Model Validation.** Your Poisson model says CVEs arrive at a constant rate. They don't. PPCs detect the clustering your model misses.
- **Gibbs and Metropolis → from scratch on network traffic.** Correlated flow features (duration × bytes). Every modern sampler descends from these two algorithms.

Nine notebooks. All runnable in Livebook. No GPU required. Five vendored datasets under 20KB.

**But the point is not cybersecurity.**

The point is that this works for any field. Manufacturing quality engineers should learn Bayesian SPC from SPC examples, not coin flips. Actuaries should learn hierarchical models from claims data, not SAT coaching. Agronomists should learn spatial models from crop yields, not abstract Gaussian processes.

The theory belongs to everyone. The examples should belong to the reader. A familiar domain is not a crutch — it is a **carrier wave** for the signal. You don't lower the math. You raise the context. When the reader already knows what a false positive costs or why a lockout threshold matters, all their attention goes to the new idea — the posterior, the credible interval, the decision boundary. The domain does the motivational work so the theory doesn't have to.

The first probability textbooks were written for artillerists because that's who was paying. We are no longer constrained by who is paying. We are constrained by who writes the examples. That constraint just fell.

Notebooks: [LINK TO REPO notebooks/bda-cyber/]

Built with eXMC (Elixir probabilistic programming on the BEAM). Original BDA3 demos by @avehtari (BSD-3). Grateful to Gelman, Vehtari, and the BDA3 authors for pedagogy worth translating — twice.

#BayesianStatistics #Cybersecurity #MachineLearning #Education #ProbabilisticProgramming

---
