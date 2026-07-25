# CSE race — measures what in-loop common-subexpression elimination does to a
# multi-RV custom-synth chain shader. Same model, rendered twice: CSE off, CSE on.
#
#   NX_VULKAN_PATH=... mix run bench/cse_race.exs
#
# Reports, for each mode: emitted shader size, glslangValidator compile time,
# and (if a Vulkan/EXLA compiler is available) a sampled posterior mean so you
# can see the two shaders are numerically identical — CSE is a pure win.
alias Exmc.Builder
alias Exmc.Dist.{Normal, HalfCauchy, Custom}

t = fn x -> Nx.tensor(x, type: :f64) end

# Normal log-density (no normalizing constant beyond the -log(sigma) term),
# vectorized over the observation vector.
nlpdf = fn obs, mu, sigma ->
  z = Nx.divide(Nx.subtract(obs, mu), sigma)
  Nx.subtract(Nx.subtract(Nx.multiply(t.(-0.5), Nx.multiply(z, z)), Nx.log(sigma)), t.(0.9189385332046727))
end

# A 3-regime Gaussian-mixture likelihood over N observations. The per-obs
# log-sum-exp shares a softmax denominator and per-regime terms that the tree
# emitter re-emits thousands of times — exactly the redundancy CSE targets.
build = fn n ->
  :rand.seed(:exsss, 123)
  returns = for _ <- 1..n, do: 0.0005 + :rand.normal() * 0.01

  ir =
    Builder.new_ir()
    |> Builder.data(Nx.tensor(returns, type: :f64))
    |> Builder.rv("mu_trend", Normal, %{mu: t.(0.001), sigma: t.(0.01)})
    |> Builder.rv("sigma_trend", HalfCauchy, %{scale: t.(0.02)})
    |> Builder.rv("sigma_mr", HalfCauchy, %{scale: t.(0.02)})
    |> Builder.rv("sigma_vol", HalfCauchy, %{scale: t.(0.05)})
    |> Builder.rv("logit_w1", Normal, %{mu: t.(0.0), sigma: t.(1.0)})
    |> Builder.rv("logit_w2", Normal, %{mu: t.(0.0), sigma: t.(1.0)})

  logpdf_fn = fn _x, p ->
    obs = p.__obs_data
    exp_w1 = Nx.exp(Nx.min(p.logit_w1, t.(10.0)))
    exp_w2 = Nx.exp(Nx.min(p.logit_w2, t.(10.0)))
    z = Nx.add(Nx.add(t.(1.0), exp_w1), exp_w2)
    log_w0 = Nx.subtract(t.(0.0), Nx.log(z))
    log_w1 = Nx.subtract(p.logit_w1, Nx.log(z))
    log_w2 = Nx.subtract(p.logit_w2, Nx.log(z))
    a = Nx.add(nlpdf.(obs, p.mu_trend, Nx.max(p.sigma_trend, t.(1.0e-8))), log_w0)
    b = Nx.add(nlpdf.(obs, t.(0.0), Nx.max(p.sigma_mr, t.(1.0e-8))), log_w1)
    c = Nx.add(nlpdf.(obs, t.(0.0), Nx.max(p.sigma_vol, t.(1.0e-8))), log_w2)
    m = Nx.max(Nx.max(a, b), c)

    lse =
      Nx.add(
        m,
        Nx.log(
          Nx.add(
            Nx.add(Nx.exp(Nx.subtract(a, m)), Nx.exp(Nx.subtract(b, m))),
            Nx.exp(Nx.subtract(c, m))
          )
        )
      )

    Nx.sum(lse)
  end

  dist = Custom.new(logpdf_fn)

  params = %{
    mu_trend: "mu_trend", sigma_trend: "sigma_trend", sigma_mr: "sigma_mr",
    sigma_vol: "sigma_vol", logit_w1: "logit_w1", logit_w2: "logit_w2",
    __obs_data: "__obs_data"
  }

  ir
  |> Custom.rv("returns_lik", dist, params)
  |> then(&Builder.obs(&1, "returns_obs", "returns_lik", t.(0.0)))
end

ir = build.(30)
{:ok, comps} = Exmc.NUTS.CustomSynth.extract_components(ir)

glslang = System.find_executable("glslangValidator")

IO.puts("\n=== CSE race: 3-regime mixture, 30 obs ===")
IO.puts(
  String.pad_trailing("mode", 9) <>
    String.pad_trailing("GLSL bytes", 13) <>
    String.pad_trailing("SPIR-V bytes", 15) <> "glslang -V"
)

results =
  for cse <- [false, true] do
    Application.put_env(:exmc, :glsl_cse, cse)
    {:ok, glsl} = Exmc.NUTS.CustomSynth.MultiRvCustomSpec.render(comps)
    path = "/tmp/cse_race_#{cse}.comp"
    File.write!(path, glsl)

    {compile_ms, spv_bytes} =
      if glslang do
        spv = "/tmp/cse_race_#{cse}.spv"
        # -V generates the real SPIR-V binary (the artifact the GPU driver
        # ingests), so this is the representative compile cost — not the much
        # cheaper -S-only front-end validation.
        t0 = System.monotonic_time(:millisecond)
        {_out, _code} = System.cmd(glslang, ["-V", "-S", "comp", path, "-o", spv], stderr_to_stdout: true)
        dt = System.monotonic_time(:millisecond) - t0
        {dt, (File.exists?(spv) && File.stat!(spv).size) || nil}
      else
        {nil, nil}
      end

    label = if cse, do: "CSE on", else: "CSE off"
    ct = if compile_ms, do: "#{compile_ms}ms", else: "(no glslang)"
    sp = if spv_bytes, do: "#{spv_bytes}", else: "-"
    IO.puts(String.pad_trailing(label, 9) <> String.pad_trailing("#{byte_size(glsl)}", 13) <> String.pad_trailing(sp, 15) <> ct)
    {byte_size(glsl), spv_bytes, compile_ms}
  end

[{off_g, off_s, off_c}, {on_g, on_s, on_c}] = results
IO.puts("\nGLSL source: #{Float.round(off_g / on_g, 1)}x smaller with CSE (#{off_g} -> #{on_g} bytes)")

if off_s && on_s,
  do: IO.puts("SPIR-V:      #{Float.round(off_s / on_s, 1)}x smaller (#{off_s} -> #{on_s} bytes)")

if off_c && on_c && on_c > 0,
  do: IO.puts("compile:     #{Float.round(off_c / on_c, 1)}x faster glslang -V (#{off_c}ms -> #{on_c}ms)")
