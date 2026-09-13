# The seven race models, eXMC side (docs/PYMC_RACE_PLAN.md).
#
# Two variants per model, because the race publishes the graph-shape cost:
#
#   :vector  -- written the idiomatic exmc way today: `shape: {n}` vector RVs,
#               and Custom likelihoods capturing their data and using Nx.dot.
#   :scalar  -- one scalar RV per coordinate joined by string references, the
#               form February's exmc builders used (phd.git benchmark/
#               race_three_way.exs, benchmark_exmc.exs @ c579dba).
#
# simple, medium and stress have no vector RVs, so both variants are the same
# model and build the same IR.
#
# Model identity with bench/pymc_race/models.py is not assumed, it is gated:
# parity.exs compares these log densities against PyMC's at 200 points per
# model. Known, deliberate differences in PARAMETER SPACE (not in the model):
# exmc samples HalfNormal through softplus, PyMC through log.
#
# `build(name, variant, data)` returns `%{ir: ir, map: map}`, where `map` sends
# every exmc free-RV id to `{pymc_var_name, :all | index}`, so draws and parity
# points convert between the two layouts.

defmodule PymcRace.Models do
  alias Exmc.{Builder, Dist}

  @models ~w(simple medium stress eight_schools funnel logistic sv)
  def models, do: @models

  defp t(v), do: Nx.tensor(v, type: :f64)

  def build("simple", _variant, data) do
    y = t(data["simple"]["y"])

    ir =
      Builder.new_ir()
      |> Builder.rv("mu", Dist.Normal, %{mu: t(0.0), sigma: t(10.0)})
      |> Builder.rv("sigma", Dist.Exponential, %{lambda: t(1.0)})
      |> Builder.rv("y", Dist.Normal, %{mu: "mu", sigma: "sigma"})
      |> Builder.obs("y_obs", "y", y)

    %{ir: ir, map: identity(~w(mu sigma))}
  end

  def build("medium", _variant, data) do
    ir =
      Builder.new_ir()
      |> Builder.rv("mu_global", Dist.Normal, %{mu: t(0.0), sigma: t(10.0)})
      |> Builder.rv("sigma_global", Dist.Exponential, %{lambda: t(1.0)})
      |> Builder.rv("alpha", Dist.Normal, %{mu: "mu_global", sigma: "sigma_global"})
      |> Builder.rv("beta", Dist.Normal, %{mu: "mu_global", sigma: "sigma_global"})
      |> Builder.rv("sigma_obs", Dist.Exponential, %{lambda: t(2.0)})
      |> Builder.rv("y_a", Dist.Normal, %{mu: "alpha", sigma: "sigma_obs"})
      |> Builder.obs("y_a_obs", "y_a", t(data["medium"]["y_a"]))
      |> Builder.rv("y_b", Dist.Normal, %{mu: "beta", sigma: "sigma_obs"})
      |> Builder.obs("y_b_obs", "y_b", t(data["medium"]["y_b"]))

    %{ir: ir, map: identity(~w(mu_global sigma_global alpha beta sigma_obs))}
  end

  def build("stress", _variant, data) do
    ir =
      Builder.new_ir()
      |> Builder.rv("mu_pop", Dist.Normal, %{mu: t(0.0), sigma: t(10.0)})
      |> Builder.rv("sigma_pop", Dist.Exponential, %{lambda: t(0.5)})

    ir =
      Enum.reduce(1..3, ir, fn j, acc ->
        acc
        |> Builder.rv("group_#{j}", Dist.Normal, %{mu: "mu_pop", sigma: "sigma_pop"})
        |> Builder.rv("noise_#{j}", Dist.Exponential, %{lambda: t(1.0)})
        |> Builder.rv("y_#{j}", Dist.Normal, %{mu: "group_#{j}", sigma: "noise_#{j}"})
        |> Builder.obs("y_#{j}_obs", "y_#{j}", t(data["stress"]["y_#{j}"]))
      end)

    ids = ~w(mu_pop sigma_pop group_1 group_2 group_3 noise_1 noise_2 noise_3)
    %{ir: ir, map: identity(ids)}
  end

  def build("eight_schools", :vector, data) do
    es = data["eight_schools"]

    ir =
      Builder.new_ir()
      |> Builder.rv("mu", Dist.Normal, %{mu: t(0.0), sigma: t(5.0)})
      |> Builder.rv("tau", Dist.HalfNormal, %{sigma: t(5.0)})
      |> Builder.rv("theta", Dist.Normal, %{mu: "mu", sigma: "tau"}, shape: {8})
      |> Builder.rv(
        "y",
        Dist.Normal,
        %{mu: "theta", sigma: t(es["sigma"] |> Enum.map(&(&1 * 1.0)))},
        shape: {8}
      )
      |> Builder.obs("y_obs", "y", t(es["y"] |> Enum.map(&(&1 * 1.0))))

    %{ir: ir, map: identity(~w(mu tau theta))}
  end

  def build("eight_schools", :scalar, data) do
    es = data["eight_schools"]

    ir =
      Builder.new_ir()
      |> Builder.rv("mu", Dist.Normal, %{mu: t(0.0), sigma: t(5.0)})
      |> Builder.rv("tau", Dist.HalfNormal, %{sigma: t(5.0)})

    ir =
      Enum.reduce(1..8, ir, fn j, acc ->
        acc
        |> Builder.rv("theta_#{j}", Dist.Normal, %{mu: "mu", sigma: "tau"})
        |> Builder.rv("y_#{j}", Dist.Normal, %{
          mu: "theta_#{j}",
          sigma: t(Enum.at(es["sigma"], j - 1) * 1.0)
        })
        |> Builder.obs("y_#{j}_obs", "y_#{j}", t(Enum.at(es["y"], j - 1) * 1.0))
      end)

    map =
      Map.merge(identity(~w(mu tau)), Map.new(1..8, fn j -> {"theta_#{j}", {"theta", j - 1}} end))

    %{ir: ir, map: map}
  end

  # x ~ Normal(0, exp(y/2)): the scale is an expression of another RV, which a
  # param map cannot name, so both variants use a Custom density. No clamp on
  # y/2 (February's exmc builder clamped it to [-20, 20], which is a different
  # density at the extremes).
  def build("funnel", variant, _data) do
    funnel =
      Dist.Custom.new(fn x, p ->
        half_y = Nx.divide(p.y_val, 2.0)
        z = Nx.divide(x, Nx.exp(half_y))
        Nx.sum(Nx.subtract(Nx.multiply(-0.5, Nx.multiply(z, z)), half_y))
      end)

    ir = Builder.new_ir() |> Builder.rv("y", Dist.Normal, %{mu: t(0.0), sigma: t(3.0)})

    case variant do
      :vector ->
        %{
          ir: Dist.Custom.rv(ir, "x", funnel, %{y_val: "y"}, shape: {9}),
          map: identity(~w(y x))
        }

      :scalar ->
        ir =
          Enum.reduce(1..9, ir, fn i, acc ->
            Dist.Custom.rv(acc, "x_#{i}", funnel, %{y_val: "y"})
          end)

        %{ir: ir, map: Map.merge(identity(~w(y)), Map.new(1..9, &{"x_#{&1}", {"x", &1 - 1}}))}
    end
  end

  # Bernoulli(logit_p = alpha + X beta), as a Custom likelihood capturing X and
  # y: sum(y * eta - softplus(eta)), softplus written overflow-safe.
  def build("logistic", variant, data) do
    lg = data["logistic"]
    x = t(lg["X"])
    y = t(Enum.map(lg["y"], &(&1 * 1.0)))
    p = elem(Nx.shape(x), 1)

    bern = fn eta ->
      softplus = Nx.add(Nx.max(eta, 0.0), Nx.log1p(Nx.exp(Nx.negate(Nx.abs(eta)))))
      Nx.sum(Nx.subtract(Nx.multiply(y, eta), softplus))
    end

    ir = Builder.new_ir() |> Builder.rv("alpha", Dist.Normal, %{mu: t(0.0), sigma: t(10.0)})

    case variant do
      :vector ->
        lik = Dist.Custom.new(fn _x, q -> bern.(Nx.add(q.alpha, Nx.dot(x, q.beta))) end)

        ir =
          ir
          |> Builder.rv("beta", Dist.Normal, %{mu: t(0.0), sigma: t(10.0)}, shape: {p})
          |> Dist.Custom.rv("Y", lik, %{alpha: "alpha", beta: "beta"})
          |> Builder.obs("Y_obs", "Y", y)

        %{ir: ir, map: identity(~w(alpha beta))}

      :scalar ->
        cols = for j <- 0..(p - 1), do: x[[.., j]]
        keys = for j <- 1..p, do: String.to_atom("beta_#{j}")

        lik =
          Dist.Custom.new(fn _x, q ->
            eta =
              Enum.zip(cols, keys)
              |> Enum.reduce(q.alpha, fn {col, k}, acc ->
                Nx.add(acc, Nx.multiply(col, Map.fetch!(q, k)))
              end)

            bern.(eta)
          end)

        ir =
          Enum.reduce(1..p, ir, fn j, acc ->
            Builder.rv(acc, "beta_#{j}", Dist.Normal, %{mu: t(0.0), sigma: t(10.0)})
          end)

        refs = Map.new(1..p, fn j -> {String.to_atom("beta_#{j}"), "beta_#{j}"} end)

        ir =
          ir
          |> Dist.Custom.rv("Y", lik, Map.put(refs, :alpha, "alpha"))
          |> Builder.obs("Y_obs", "Y", y)

        %{
          ir: ir,
          map: Map.merge(identity(~w(alpha)), Map.new(1..p, &{"beta_#{&1}", {"beta", &1 - 1}}))
        }
    end
  end

  # Log-volatility random walk with StudentT returns, scale exp(s). The
  # likelihood's scale is an expression of s, so it is a Custom capturing the
  # returns and calling exmc's own StudentT density.
  def build("sv", variant, data) do
    sv = data["sv"]
    returns = t(sv["returns"])
    big_t = sv["T"]

    ir =
      Builder.new_ir()
      |> Builder.rv("sigma", Dist.Exponential, %{lambda: t(50.0)})
      |> Builder.rv("nu", Dist.Exponential, %{lambda: t(0.1)})

    studentt = fn nu, s ->
      Dist.StudentT.logpdf(returns, %{df: nu, loc: t(0.0), scale: Nx.exp(s)}) |> Nx.sum()
    end

    case variant do
      :vector ->
        lik = Dist.Custom.new(fn _x, q -> studentt.(q.nu, q.s) end)

        ir =
          ir
          |> Builder.rv("s", Dist.GaussianRandomWalk, %{sigma: "sigma"}, shape: {big_t})
          |> Dist.Custom.rv("R", lik, %{nu: "nu", s: "s"})
          |> Builder.obs("R_obs", "R", returns)

        %{ir: ir, map: identity(~w(sigma nu s))}

      :scalar ->
        keys = for i <- 1..big_t, do: String.to_atom("s_#{i}")

        lik =
          Dist.Custom.new(fn _x, q ->
            studentt.(q.nu, Nx.stack(Enum.map(keys, &Map.fetch!(q, &1))))
          end)

        ir =
          Enum.reduce(1..big_t, ir, fn i, acc ->
            mu = if i == 1, do: t(0.0), else: "s_#{i - 1}"
            Builder.rv(acc, "s_#{i}", Dist.Normal, %{mu: mu, sigma: "sigma"})
          end)

        refs = Map.new(1..big_t, fn i -> {String.to_atom("s_#{i}"), "s_#{i}"} end)

        ir =
          ir
          |> Dist.Custom.rv("R", lik, Map.put(refs, :nu, "nu"))
          |> Builder.obs("R_obs", "R", returns)

        %{
          ir: ir,
          map: Map.merge(identity(~w(sigma nu)), Map.new(1..big_t, &{"s_#{&1}", {"s", &1 - 1}}))
        }
    end
  end

  defp identity(ids), do: Map.new(ids, &{&1, {&1, :all}})
end
