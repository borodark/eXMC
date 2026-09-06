# Leaf-level differential: synthesised chain shader vs host leapfrog.
#
# NEXT.md §3 / exmc docs/OPEN_VULKAN_OBSERVED_MODEL.md "Next experiment".
# Fix q0, p0, eps, inv_mass, K. Dispatch leapfrog_chain_synth_f64, read back
# q_chain / p_chain / grad_chain / logp_chain. Run the same K leapfrog steps on
# the host via Compiler.value_and_grad/1. Compare all four element-wise.

alias Exmc.{Builder, Compiler}
alias Exmc.Dist.Normal
alias Exmc.NUTS.{CustomSynth, Leapfrog}
alias Exmc.NUTS.Vulkan.Dispatch

f64 = [type: :f64, backend: Nx.BinaryBackend]

# q/p/grad are {1}-shaped here, logp is scalar — take the first element either way.
num = fn t -> t |> Nx.to_flat_list() |> hd() end

# NOTE ON THE SIGMAS. They are 1.0 / 2.0 / 3.0, not all 1.0 as in the model
# that first showed the defect, and that is load-bearing rather than cosmetic.
#
# The fix attributes each observed node a SLICE of the observation buffer, and
# marker i is matched to node i positionally. If that correspondence were ever
# permuted — the gradient markers come out of `Nx.Defn.grad`, which owes nobody
# forward order — the offsets would land on the wrong observations. With every
# node identical the permuted answer is bit-for-bit the correct one and this
# harness would pass while the code was wrong. Distinct sigmas make any
# permutation change the numbers.
ir =
  Builder.new_ir()
  |> Builder.rv("mu", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(10.0)})
  |> Builder.rv("x1", Normal, %{mu: "mu", sigma: Nx.tensor(1.0)})
  |> Builder.obs("x1_obs", "x1", Nx.tensor(4.0))
  |> Builder.rv("x2", Normal, %{mu: "mu", sigma: Nx.tensor(2.0)})
  |> Builder.obs("x2_obs", "x2", Nx.tensor(3.8))
  |> Builder.rv("x3", Normal, %{mu: "mu", sigma: Nx.tensor(3.0)})
  |> Builder.obs("x3_obs", "x3", Nx.tensor(4.2))

{:ok, meta} = CustomSynth.synthesise(ir)
# 7-tuple since captures moved to the extras SSBO. This file destructured the
# old 6-tuple for a day after that change: every TEST was updated, this bench
# was not, and `mix test` never runs it. It is the ONLY harness in the repo
# that dispatches a synthesised shader and compares logp/grad element-wise
# against the host -- so the one instrument that could have caught the
# zero-likelihood reduce-bound defect was itself un-runnable, on the very
# commit that made that defect reachable. Grep the whole repo, not test/.
{:synthesised, sha, layout, push_spec, spv_path, obs_bin, _captures_bin} = meta

IO.puts("""
device : #{inspect(Nx.Vulkan.NativeV.device_name())}
sha    : #{String.slice(sha, 0, 16)}
layout : #{inspect(layout)}
spv    : #{spv_path}
obs    : #{byte_size(obs_bin)} bytes -> #{inspect(Nx.to_flat_list(Nx.from_binary(obs_bin, :f64, backend: Nx.BinaryBackend)))}
push   : n_obs=#{push_spec.n_obs} priors=#{inspect(push_spec.priors)}
""")

d = 1
k = 32
{vag_fn, _pm} = Compiler.value_and_grad(ir)

# Independent analytic check of the host gradient, so a host/GPU agreement
# cannot both be wrong in the same direction.
#   logp(mu) = -0.5 (mu/10)^2 - 0.5 sum ((x_i - mu)/s_i)^2 + const
#   dlogp/dmu = -mu/100 + sum (x_i - mu)/s_i^2
analytic_grad = fn mu ->
  -mu / 100.0 +
    Enum.sum(for {x, sd} <- [{4.0, 1.0}, {3.8, 2.0}, {4.2, 3.0}], do: (x - mu) / (sd * sd))
end

host_chain = fn q0, p0, eps ->
  {lp0, g0} = vag_fn.(q0)

  {rows, _} =
    Enum.map_reduce(1..k, {q0, p0, g0, lp0}, fn _i, {q, p, g, _lp} ->
      {qn, pn, lpn, gn} =
        Leapfrog.step(vag_fn, q, p, g, eps, Nx.tensor([1.0], f64))

      {{qn, pn, gn, lpn}, {qn, pn, gn, lpn}}
    end)

  rows
end

cmp = fn label, gpu_t, host_vals ->
  gpu = Nx.to_flat_list(gpu_t)
  host = List.flatten(host_vals)

  # Tolerance is RELATIVE and set at 1e-6, not f64 epsilon, and that is a
  # statement about the shader rather than slack. The emitted GLSL carries its
  # distribution constants at f32 (log(2*pi) appears as 1.8378770351409912) and
  # the observation buffer is packed from f32 tensors, so 3.8 arrives as
  # 3.799999952316284. ~1e-7 relative is the floor this path can reach; the
  # defect this harness was built for was 3x, six orders of magnitude above it.
  diffs =
    Enum.zip(gpu, host)
    |> Enum.map(fn {a, b} -> abs(a - b) / max(1.0, abs(b)) end)

  max = Enum.max(diffs)

  first =
    diffs
    |> Enum.with_index()
    |> Enum.find(fn {dv, _i} -> dv > 1.0e-6 end)

  msg =
    case first do
      nil -> "agrees (max relative Δ = #{max})"
      {dv, i} ->
        "DIVERGES at step #{i}: gpu=#{Enum.at(gpu, i)} host=#{Enum.at(host, i)} " <>
          "relΔ=#{dv} (max relative Δ = #{max})"
    end

  IO.puts("  #{String.pad_trailing(label, 6)} #{msg}")
  first == nil
end

run = fn eps, q0v, p0v ->
  IO.puts("\n=== eps = #{eps}, q0 = #{q0v}, p0 = #{p0v}, K = #{k} ===")
  q0 = Nx.tensor([q0v], f64)
  p0 = Nx.tensor([p0v], f64)
  inv_mass = Nx.tensor([1.0], f64)

  {lp0, g0} = vag_fn.(q0)

  IO.puts(
    "  host at q0: logp=#{num.(lp0)} grad=#{num.(g0)} " <>
      "(analytic grad=#{analytic_grad.(q0v)})"
  )

  # Dispatch order is {q, p, logp, grad} — see Dispatch.bins_to_chain_tensors/4.
  {q_c, p_c, lp_c, g_c} = Dispatch.chain(meta, d, eps, inv_mass, q0, p0, k, 1)

  host = host_chain.(q0, p0, eps)
  hq = Enum.map(host, fn {q, _, _, _} -> num.(q) end)
  hp = Enum.map(host, fn {_, p, _, _} -> num.(p) end)
  hg = Enum.map(host, fn {_, _, g, _} -> num.(g) end)
  hlp = Enum.map(host, fn {_, _, _, lp} -> num.(lp) end)

  ok_q = cmp.("q", q_c, hq)
  ok_p = cmp.("p", p_c, hp)
  ok_g = cmp.("grad", g_c, hg)
  ok_lp = cmp.("logp", lp_c, hlp)

  # logp may legitimately differ by a constant normaliser between the two
  # implementations; what must match is the SHAPE along the trajectory,
  # because that is what the Metropolis ratio consumes.
  gl = Nx.to_flat_list(lp_c)
  offs = Enum.zip(gl, hlp) |> Enum.map(fn {a, b} -> a - b end)
  spread = Enum.max(offs) - Enum.min(offs)

  IO.puts(
    "  logp offset: first=#{Enum.at(offs, 0)} spread over trajectory=#{spread}" <>
      if(spread < 1.0e-9, do: " (constant -> ratio-equivalent)", else: " (NOT constant)")
  )

  IO.puts("  first 4 gpu  q: #{inspect(Enum.take(Nx.to_flat_list(q_c), 4))}")
  IO.puts("  first 4 host q: #{inspect(Enum.take(hq, 4))}")

  {ok_q, ok_p, ok_g, ok_lp}
end

# Small eps first: if the math is right, this MUST agree to ~1e-15. Then the
# adapted eps from the bug report, where the trajectory is 2-sigma per step and
# any real discrepancy is amplified.
run.(0.05, 0.5, 1.0)
run.(1.1391216000810296, 0.5, 1.0)
run.(1.1391216000810296, 3.99, 0.1)
