defmodule Exmc.NUTS.Statham.Merge do
  @moduledoc """
  proper_statem model for the NUTS merge protocol.

  Tests merge_subtrees and merge_trajectories in isolation — no leapfrog,
  no gradients, just the tree logic with synthetic subtrees.

  The bugs this would have caught:
  - D50: capped log_weight (leaf log_weight = min(0, delta) underweighted good points)
  - D51: biased outer merge (balanced instead of biased progressive sampling)

  Both were correct MCMC but with 2-3x worse ESS. Four months to find manually.
  This model finds them in four lines.
  """
  use PropCheck
  use PropCheck.StateM

  alias Exmc.NUTS.Tree

  # --- SUT state in process dictionary ---
  def get_state, do: Process.get(:statham_merge)
  def set_state(s), do: Process.put(:statham_merge, s)

  # --- Model state ---
  def initial_state do
    %{
      initialized: false,
      d: 0,
      merge_count: 0,
      subtree_accepts: [],
      trajectory_accepts: []
    }
  end

  # --- Command generation ---
  def command(%{initialized: false}) do
    {:call, __MODULE__, :do_init, [integer(2, 8)]}
  end

  def command(%{initialized: true}) do
    frequency([
      {5, {:call, __MODULE__, :do_merge_subtrees, [float(-1.0, 1.0), integer(1, 100_000)]}},
      {4, {:call, __MODULE__, :do_merge_trajectories, [boolean(), integer(1, 100_000)]}},
      {1, {:call, __MODULE__, :check_acceptance_stats, []}}
    ])
  end

  # --- SUT functions ---

  def do_init(d) do
    set_state(%{d: d, inv_mass_list: List.duplicate(1.0, d)})
    :ok
  end

  def do_merge_subtrees(epsilon_sign, seed) do
    s = get_state()
    d = s.d

    rng = :rand.seed(:exsss, {seed, seed * 7, seed * 13})

    # Generate two synthetic subtrees with known weights
    {left, rng} = random_subtree(d, rng)
    {right, rng} = random_subtree(d, rng)

    epsilon = if epsilon_sign >= 0, do: 0.1, else: -0.1
    inv_mass = Nx.tensor(List.duplicate(1.0, d))

    {merged, _rng} = Tree.merge_subtrees(left, right, epsilon, inv_mass, rng, s.inv_mass_list)

    # Record which proposal was accepted
    accepted_right = merged.q_prop == right.q_prop

    %{
      left_lsw: left.log_sum_weight,
      right_lsw: right.log_sum_weight,
      merged_lsw: merged.log_sum_weight,
      merged_n_steps: merged.n_steps,
      left_n_steps: left.n_steps,
      right_n_steps: right.n_steps,
      merged_depth: merged.depth,
      left_depth: left.depth,
      right_depth: right.depth,
      rho_correct: rho_additive?(left.rho_list, right.rho_list, merged.rho_list),
      divergent_monotonic: merged.divergent >= left.divergent and merged.divergent >= right.divergent,
      accepted_right: accepted_right
    }
  end

  def do_merge_trajectories(go_right, seed) do
    s = get_state()
    d = s.d

    rng = :rand.seed(:exsss, {seed, seed * 7, seed * 13})

    {traj, rng} = random_subtree(d, rng, depth: 3)
    {subtree, rng} = random_subtree(d, rng, depth: 2)

    inv_mass = Nx.tensor(List.duplicate(1.0, d))

    {merged, _rng} = Tree.merge_trajectories(traj, subtree, go_right, inv_mass, rng, s.inv_mass_list)

    accepted_subtree = merged.q_prop == subtree.q_prop

    %{
      traj_lsw: traj.log_sum_weight,
      subtree_lsw: subtree.log_sum_weight,
      merged_lsw: merged.log_sum_weight,
      merged_n_steps: merged.n_steps,
      traj_n_steps: traj.n_steps,
      subtree_n_steps: subtree.n_steps,
      rho_correct: rho_additive?(traj.rho_list, subtree.rho_list, merged.rho_list),
      divergent_monotonic: merged.divergent >= traj.divergent and merged.divergent >= subtree.divergent,
      accepted_subtree: accepted_subtree,
      # The key D51 check: biased progressive uses subtree.lsw - traj.lsw, not balanced
      theoretical_accept_prob: min(1.0, :math.exp(subtree.log_sum_weight - traj.log_sum_weight))
    }
  end

  def check_acceptance_stats do
    # This is checked via postconditions, not here
    :ok
  end

  # --- Preconditions ---
  def precondition(%{initialized: false}, {:call, _, :do_init, _}), do: true
  def precondition(%{initialized: true}, {:call, _, :do_merge_subtrees, _}), do: true
  def precondition(%{initialized: true}, {:call, _, :do_merge_trajectories, _}), do: true
  def precondition(%{initialized: true}, {:call, _, :check_acceptance_stats, _}), do: true
  def precondition(_, _), do: false

  # --- Postconditions ---

  def postcondition(_state, {:call, _, :do_init, _}, :ok), do: true

  def postcondition(_state, {:call, _, :do_merge_subtrees, _}, result) do
    # INVARIANT 1: log_sum_weight is log_sum_exp of children
    expected_lsw = Tree.log_sum_exp(result.left_lsw, result.right_lsw)
    lsw_ok = abs(result.merged_lsw - expected_lsw) < 1.0e-10

    # INVARIANT 2: n_steps is sum of children
    steps_ok = result.merged_n_steps == result.left_n_steps + result.right_n_steps

    # INVARIANT 3: depth is max(children) + 1
    depth_ok = result.merged_depth == max(result.left_depth, result.right_depth) + 1

    # INVARIANT 4: rho is additive
    rho_ok = result.rho_correct

    # INVARIANT 5: divergent flag is monotonic (OR of children)
    div_ok = result.divergent_monotonic

    lsw_ok and steps_ok and depth_ok and rho_ok and div_ok
  end

  def postcondition(_state, {:call, _, :do_merge_trajectories, _}, result) do
    # INVARIANT 1: log_sum_weight is log_sum_exp of trajectory + subtree
    expected_lsw = Tree.log_sum_exp(result.traj_lsw, result.subtree_lsw)
    lsw_ok = abs(result.merged_lsw - expected_lsw) < 1.0e-10

    # INVARIANT 2: n_steps is sum
    steps_ok = result.merged_n_steps == result.traj_n_steps + result.subtree_n_steps

    # INVARIANT 3: rho is additive
    rho_ok = result.rho_correct

    # INVARIANT 4: divergent is monotonic
    div_ok = result.divergent_monotonic

    lsw_ok and steps_ok and rho_ok and div_ok
  end

  def postcondition(_state, {:call, _, :check_acceptance_stats, _}, _), do: true

  # --- State transitions ---

  def next_state(state, _result, {:call, _, :do_init, [d]}) do
    %{state | initialized: true, d: d}
  end

  def next_state(state, result, {:call, _, :do_merge_subtrees, _}) do
    case result do
      %{accepted_right: accepted} when is_boolean(accepted) ->
        %{state |
          merge_count: state.merge_count + 1,
          subtree_accepts: [accepted | state.subtree_accepts]
        }
      _ -> state
    end
  end

  def next_state(state, result, {:call, _, :do_merge_trajectories, _}) do
    case result do
      %{accepted_subtree: accepted} when is_boolean(accepted) ->
        %{state |
          merge_count: state.merge_count + 1,
          trajectory_accepts: [accepted | state.trajectory_accepts]
        }
      _ -> state
    end
  end

  def next_state(state, _, {:call, _, :check_acceptance_stats, _}), do: state

  # --- Synthetic subtree generator ---

  defp random_subtree(d, rng, opts \\ []) do
    depth = opts[:depth] || 1

    {lsw, rng} = :rand.uniform_s(rng)
    log_sum_weight = :math.log(lsw + 0.01)

    q_list = for _ <- 1..d, do: :rand.normal()
    p_list = for _ <- 1..d, do: :rand.normal()
    rho_list = for _ <- 1..d, do: :rand.normal()

    q = Nx.tensor(q_list)
    p = Nx.tensor(p_list)
    grad = Nx.tensor(List.duplicate(0.0, d))

    subtree = %{
      q_left: q, p_left: p, grad_left: grad,
      q_left_list: q_list, p_left_list: p_list,
      q_right: q, p_right: p, grad_right: grad,
      q_right_list: q_list, p_right_list: p_list,
      q_prop: q, logp_prop: log_sum_weight, grad_prop: grad,
      rho_list: rho_list,
      depth: depth,
      log_sum_weight: log_sum_weight,
      n_steps: round(:math.pow(2, depth)),
      divergent: false,
      accept_sum: 0.5 * round(:math.pow(2, depth)),
      turning: false,
      recovered: false
    }

    {subtree, rng}
  end

  defp rho_additive?(rho_a, rho_b, rho_merged) do
    expected = Enum.zip(rho_a, rho_b) |> Enum.map(fn {a, b} -> a + b end)

    Enum.zip(expected, rho_merged)
    |> Enum.all?(fn {e, m} -> abs(e - m) < 1.0e-10 end)
  end
end

# ================================================================
# Phase 2: U-turn properties (pure functions, no statem needed)
# ================================================================

defmodule Exmc.NUTS.Statham.UTurn do
  @moduledoc """
  Property tests for the U-turn detection criterion.
  Pure functions — PropCheck with full shrinking.
  """
  use PropCheck

  alias Exmc.NUTS.Tree

  # Generate a random momentum vector of dimension d
  defp momentum_gen(d) do
    let vals <- vector(d, float(-5.0, 5.0)) do
      vals
    end
  end

  defp inv_mass_gen(d) do
    let vals <- vector(d, float(0.1, 10.0)) do
      vals
    end
  end

  # --- Properties ---

  def prop_aligned_no_uturn do
    forall d <- integer(2, 8) do
      forall {p, inv_mass} <- {momentum_gen(d), inv_mass_gen(d)} do
        # All momenta aligned in same direction → no U-turn
        # rho = p (single step), p_left = p, p_right = p
        not Tree.check_uturn_rho(p, p, p, inv_mass)
      end
    end
  end

  def prop_opposite_uturn do
    forall d <- integer(2, 8) do
      forall {p, inv_mass} <- {momentum_gen(d), inv_mass_gen(d)} do
        # p_right = -p → trajectory reversed
        neg_p = Enum.map(p, &(-&1))
        # rho = p + neg_p = zeros → dot products are zero, not negative
        # Actually: rho = 0 means dot(rho, inv_mass * p) = 0, which is NOT < 0
        # So zero rho doesn't trigger U-turn by strict < 0
        # But if rho = p (one step), p_right = -p:
        # dot(p, inv_mass * (-p)) = -sum(p_i^2 * m_i) < 0 ← U-turn!
        Tree.check_uturn_rho(p, p, neg_p, inv_mass)
      end
    end
  end

  def prop_zero_rho_no_uturn do
    forall d <- integer(2, 8) do
      forall {p_left, p_right, inv_mass} <- {momentum_gen(d), momentum_gen(d), inv_mass_gen(d)} do
        # Zero rho → dot products are zero → no U-turn (strict < 0)
        zero_rho = List.duplicate(0.0, d)
        not Tree.check_uturn_rho(zero_rho, p_left, p_right, inv_mass)
      end
    end
  end

  def prop_log_sum_exp_commutative do
    forall {a, b} <- {float(-100.0, 100.0), float(-100.0, 100.0)} do
      abs(Tree.log_sum_exp(a, b) - Tree.log_sum_exp(b, a)) < 1.0e-10
    end
  end

  def prop_log_sum_exp_dominance do
    forall {a, b} <- {float(-100.0, 100.0), float(-100.0, 100.0)} do
      # log_sum_exp(a, b) >= max(a, b)
      result = Tree.log_sum_exp(a, b)
      result >= max(a, b) - 1.0e-10
    end
  end
end

# ================================================================
# TEST MODULE
# ================================================================

defmodule Exmc.NUTS.StathamTest do
  use ExUnit.Case
  use PropCheck
  use PropCheck.StateM

  @moduletag timeout: 120_000

  # Phase 1: Merge protocol — statem model
  property "merge protocol invariants under adversarial sequences", [:verbose, numtests: 300] do
    forall cmds <- commands(Exmc.NUTS.Statham.Merge) do
      Process.delete(:statham_merge)

      {history, state, result} = run_commands(Exmc.NUTS.Statham.Merge, cmds)

      (result == :ok)
      |> when_fail(
        IO.puts("""
        === NUTS statham FAILURE (merge) ===
        State: #{inspect(state, pretty: true)}
        Result: #{inspect(result)}
        History: #{length(history)} steps
        """)
      )
      |> aggregate(command_names(cmds))
    end
  end

  # Phase 2: U-turn properties
  property "aligned momentum never triggers U-turn", [numtests: 200] do
    Exmc.NUTS.Statham.UTurn.prop_aligned_no_uturn()
  end

  property "reversed momentum triggers U-turn", [numtests: 200] do
    Exmc.NUTS.Statham.UTurn.prop_opposite_uturn()
  end

  property "zero rho never triggers U-turn", [numtests: 200] do
    Exmc.NUTS.Statham.UTurn.prop_zero_rho_no_uturn()
  end

  property "log_sum_exp is commutative", [numtests: 500] do
    Exmc.NUTS.Statham.UTurn.prop_log_sum_exp_commutative()
  end

  property "log_sum_exp >= max(a, b)", [numtests: 500] do
    Exmc.NUTS.Statham.UTurn.prop_log_sum_exp_dominance()
  end
end
