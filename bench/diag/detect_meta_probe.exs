# Which models reach exmc's fused chain shader, and why the others do not.
#
# Findings (exmc cbc858ee7):
#
#   A  vector RV shape:{2}, Custom + capture   -> :unsupported
#   B  two scalar RVs,      Custom + capture   -> {:error, GLSL compile failed}
#   C  one scalar RV,       Custom, no capture -> {:unsupported, :empty_obs_axis_reduction}
#   D  one scalar RV,       no Custom          -> {:ok, {:synthesised, ...}}   (control)
#   E  B, but resid*resid instead of pow(_, 2) -> {:ok, {:synthesised, ...}}
#
# Two independent blockers, and captures are neither of them -- E shows that
# Custom-plus-captures reaches the shader fine:
#
#   1. `Nx.pow` emits GLSL `pow` with no matching f64 overload, so the shader
#      fails to COMPILE. B vs E isolates it: the only difference is pow. This
#      is why PathMC.Compile.Exmc squares by multiplication.
#   2. A vector-valued RV (`shape: {2}`) is refused earlier, at detect, before
#      codegen runs. This is the remaining reason pathmc models miss the
#      shader -- each equation's coefficients are one vector RV.
#
# Note detect_meta/2 returns shapes outside its own @spec: `{:error, map()}`
# and `{:ok, {:synthesised, ...}}` (a 7-tuple, not a meta() map).
#
# CELL C IS DELIBERATELY ARTIFICIAL. Intercept-only is not a model anyone
# fits; it is the only way to have a Custom likelihood with NO captured
# tensor while keeping a real IR, which is what isolates captures from
# Custom. Do not "fix" it into a realistic model -- that removes the
# isolation and the probe stops discriminating.
#
# Contributed by the pathmc_ex session, which found both blockers with it.
# Updated for exmc da6ea6887: cell B now synthesises (the pow fix), so B and
# E should now AGREE. If they diverge again, pow regressed.
#
# Same posterior, two formulations:
#   A. one `shape: {2}` beta, closure does Nx.dot(x, beta)     -- the oracle
#   B. two scalar RVs beta0/beta1, closure does b0 + b1 * x
#
# Reports detect_meta/2's RAW return for each. Sampling works on both paths;
# that is the whole problem.

alias Exmc.{Builder, Dist, IR}
alias Exmc.NUTS.ChainShaderCodegen

f64 = [type: :f64]

xs = Enum.map(1..40, &(&1 / 4))
noise = [0.12, -0.31, 0.05, 0.22, -0.18, 0.31, -0.07, 0.14, -0.25, 0.09]

ys =
  xs |> Enum.with_index() |> Enum.map(fn {x, i} -> 1.0 + 2.0 * x + Enum.at(noise, rem(i, 10)) end)

xcol = Nx.tensor(xs, f64)
x = Nx.stack([Nx.broadcast(Nx.tensor(1.0, f64), {40}), xcol], axis: 1)
y = Nx.tensor(ys, f64)

sigma = 0.2
prior_sd = 5.0
norm = Nx.tensor(:math.log(sigma * :math.sqrt(2 * :math.pi())), f64)

gauss = fn observed, mu ->
  resid = Nx.subtract(observed, mu)
  Nx.sum(Nx.negate(Nx.add(norm, Nx.divide(Nx.pow(resid, 2), 2 * sigma * sigma))))
end

# --- A: one vector RV ---------------------------------------------------------
vector_ir =
  IR.new()
  |> Builder.rv("beta", Dist.Normal, %{mu: Nx.tensor(0.0, f64), sigma: Nx.tensor(prior_sd, f64)},
    shape: {2}
  )
  |> Dist.Custom.rv(
    "Y",
    Dist.Custom.new(fn observed, p -> gauss.(observed, Nx.dot(x, p.beta)) end),
    %{beta: "beta"}
  )
  |> Builder.obs("Y_obs", "Y", y)

# --- B: two scalar RVs --------------------------------------------------------
scalar_ir =
  IR.new()
  |> Builder.rv("beta0", Dist.Normal, %{mu: Nx.tensor(0.0, f64), sigma: Nx.tensor(prior_sd, f64)})
  |> Builder.rv("beta1", Dist.Normal, %{mu: Nx.tensor(0.0, f64), sigma: Nx.tensor(prior_sd, f64)})
  |> Dist.Custom.rv(
    "Y",
    Dist.Custom.new(fn observed, p ->
      gauss.(observed, Nx.add(p.beta0, Nx.multiply(p.beta1, xcol)))
    end),
    %{beta0: "beta0", beta1: "beta1"}
  )
  |> Builder.obs("Y_obs", "Y", y)

# --- C: two scalar RVs, NO captured tensor in the closure ---------------------
# Isolates "captures" from "Custom". The x column is passed as observed data by
# folding it into the response is not possible, so this one drops the covariate
# entirely: intercept-only, closure captures nothing but scalars.
no_capture_ir =
  IR.new()
  |> Builder.rv("mu", Dist.Normal, %{mu: Nx.tensor(0.0, f64), sigma: Nx.tensor(prior_sd, f64)})
  |> Dist.Custom.rv(
    "Y",
    Dist.Custom.new(fn observed, p -> gauss.(observed, p.mu) end),
    %{mu: "mu"}
  )
  |> Builder.obs("Y_obs", "Y", y)

# --- D: no Custom at all, plain Normal likelihood -----------------------------
plain_ir =
  IR.new()
  |> Builder.rv("mu", Dist.Normal, %{mu: Nx.tensor(0.0, f64), sigma: Nx.tensor(prior_sd, f64)})
  |> Builder.rv("Y", Dist.Normal, %{mu: "mu", sigma: Nx.tensor(sigma, f64)})
  |> Builder.obs("Y_obs", "Y", y)

gauss_nopow = fn observed, mu ->
  resid = Nx.subtract(observed, mu)
  Nx.sum(Nx.negate(Nx.add(norm, Nx.divide(Nx.multiply(resid, resid), 2 * sigma * sigma))))
end

nopow_ir =
  IR.new()
  |> Builder.rv("beta0", Dist.Normal, %{mu: Nx.tensor(0.0, f64), sigma: Nx.tensor(prior_sd, f64)})
  |> Builder.rv("beta1", Dist.Normal, %{mu: Nx.tensor(0.0, f64), sigma: Nx.tensor(prior_sd, f64)})
  |> Dist.Custom.rv(
    "Y",
    Dist.Custom.new(fn observed, p ->
      gauss_nopow.(observed, Nx.add(p.beta0, Nx.multiply(p.beta1, xcol)))
    end),
    %{beta0: "beta0", beta1: "beta1"}
  )
  |> Builder.obs("Y_obs", "Y", y)

for {label, ir} <- [
      {"A vector RV shape:{2}, Custom + capture", vector_ir},
      {"B two scalar RVs,      Custom + capture", scalar_ir},
      {"C one scalar RV,       Custom no capture", no_capture_ir},
      {"D one scalar RV,       no Custom        ", plain_ir},
      {"E two scalar RVs, Custom+capture, no pow", nopow_ir}
    ] do
  result =
    try do
      ChainShaderCodegen.detect_meta(ir, [])
    rescue
      e -> {:raised, e.__struct__}
    end

  summary =
    case result do
      {:ok, meta} when is_map(meta) ->
        "{:ok, meta #{inspect(Map.keys(meta))}}"

      {:synthesised, hash, vars, meta, _path, _bin, _} ->
        "{:synthesised, #{String.slice(hash, 0, 8)}.., #{inspect(vars)}, #{inspect(meta[:d])}d}"

      {:error, %{stderr: err}} ->
        first = err |> String.split("\n") |> Enum.find("", &String.contains?(&1, "ERROR"))
        "{:error, GLSL COMPILE FAILED} #{String.trim(first)}"

      other ->
        inspect(other)
    end

  IO.puts("#{label}  nodes=#{map_size(ir.nodes)}  -> #{summary}")
end
