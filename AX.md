                                                                             
  ---                                                                                                           
  What we wrote about (12 chapters): D1 (Nx), D2 (IR), D16 (:rand), variance bug, D21/D24 (lgamma/EXLA), D27
  (fused step), D32 (NCP), D28 (constrained parent bug), D35 (parallel chains), ExmcViz, live streaming, tests. 
                                                                               
  What Hitchens would have seized on that we missed or underplayed:

  1. D25: The Softplus Identity -- "The naive formula is a lie"

  log1p(exp(x)) overflows. The textbook formula is wrong for real hardware. Exmc rewrites it as max(x,0) +
  log1p(exp(-|x|)) which never overflows. Hitchens would adore this: the entire numerical computing
  establishment teaches one formula, and it doesn't work. He'd frame it as the academic-practitioner gap -- the
  people who write the textbooks don't test on BinaryBackend at x = 800.

  2. D26: Nx.to_number Returns Atoms -- "The type system lied"

  :nan, :neg_infinity, :infinity are atoms, not floats. Erlang arithmetic on them crashes with no error message
  -- just a dead process. Hitchens would frame this as Erlang's one moment of dishonesty: a function called
  to_number that sometimes returns things that are emphatically not numbers. The is_number/1 guard sprinkled
  through the sampler is a permanent monument to this betrayal.

  3. D9: Free RVs Identified by Exclusion -- "Innocence by lack of accusation"

  A random variable is free if and only if no observation points at it. This is not a positive declaration of
  freedom -- it's the absence of constraint. Hitchens (the civil libertarian) would absolutely have drawn the
  parallel to presumption of innocence and made it a chapter on how the IR's design philosophy mirrors legal
  principles.

  4. D11: Compiler Pre-dispatches at Build Time -- "The trial of closures"

  The entire logp function is closures all the way down, assembled once by walking the IR. At runtime it's pure
  Nx ops, traceable by autodiff. This is the decision that makes EXLA JIT possible -- and the one that nearly
  made it impossible (D24: captured tensors must be BinaryBackend). Hitchens would have loved the irony: the
  architecture that enabled JIT almost prevented it, because EXLA can't trace tensors that are already on EXLA.

  5. D17: Scalar Math for Adaptation, Nx for Geometry -- "Two systems of arithmetic in one sampler"

  The sampler literally uses two different number systems. Dual averaging and Welford run in Erlang native
  floats. Leapfrog and momentum run in Nx tensors. This is a pragmatic split that would offend any purist, and
  Hitchens loved offending purists. The justification is that Nx BinaryBackend adds ~100x overhead for scalar
  operations that don't benefit from tensors.

  6. The Step Size Collapse (Debugging notes) -- "The feedback loop that wasn't"

  The step size collapsed to 1e-8 because run_phase used a fixed epsilon while dual averaging tracked what it
  should be. The adaptation was running correctly in its own little world, producing perfectly good step sizes
  that nobody was using. Hitchens would have turned this into a metaphor about institutions that produce
  excellent recommendations that no one reads.

  7. D14 vs D24: The Contradiction -- "defn considered harmful, then essential"

  D14 says: no defn, plain Elixir + Nx. D24 says: actually, wrap the hot path in EXLA.jit. These decisions
  contradict each other -- and both are correct for their context. Hitchens lived for productive contradictions.
   The sampler loop is plain Elixir (D14 holds). The inner gradient is JIT-compiled (D24 overrides D14 for one
  function). The project contains its own refutation.

  8. D12 → D22: Eagerness Defeated by Hierarchy

  D12 says observation terms can be computed eagerly as constants. D22 introduces hierarchical string refs that
  make some obs terms not constant. D12 is "partially superseded." Hitchens would have noted that this is the
  pattern of every confident assumption meeting reality -- and that the document's honesty about being
  superseded is more valuable than the original decision.

  9. Memory Lesson #8 + #12: The Init Values Problem

  Without good starting points, the step-size search explores extreme unconstrained regions and math.exp
  overflows. And init values must cover ALL free RVs -- miss one and you get a KeyError. The sampler is a
  machine that requires priming, and the priming must be complete and correct. Hitchens would have seen this as
  the bootstrapping problem of all inference: you need to know approximately where the truth is before you can
  find it precisely.

  ---
  My recommendation: Chapters I (GIL), III (:rand), V (lgamma), VIII (constrained parent bug), IX (parallel
  chains) are perfect Hitchens material and should stay. But I'd swap or add:

  - D25 (softplus overflow) replaces or joins Chapter V -- "the textbook formula that doesn't work" is peak
  Hitchens
  - D26 (atoms from to_number) deserves its own chapter -- "a function that lies about its return type"
  - D14 vs D24 (the contradiction) could replace Chapter VI (fused step, which is competent engineering but not
  polemical)
  - The step-size collapse could join Chapter IV -- two feedback-loop bugs in one chapter about "systems that
  produce correct answers that nobody uses"
  - D12 → D22 (eagerness superseded) belongs in Chapter II (Memoranda) as the prime example of a decision
  honestly revising itself
  
  

VIII. The Most Dangerous Bug
I have saved the most instructive bug for its own chapter, because it embodies a principle that I believe applies well beyond software engineering: the answers that are almost right are far more dangerous than the answers that are obviously wrong.


