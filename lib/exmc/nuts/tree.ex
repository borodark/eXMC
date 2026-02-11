defmodule Exmc.NUTS.Tree do
  @moduledoc """
  NUTS binary tree building with multinomial sampling and U-turn detection.

  Implements the No-U-Turn Sampler with iterative doubling, per Hoffman & Gelman (2014)
  and Betancourt (2017) multinomial variant.

  ## Dispatch hierarchy

  Subtrees are built via one of three backends, selected per-subtree by depth
  and available acceleration:

  1. **NIF** (`build_subtree_nif`) — Rust processes pre-computed leapfrog states
     including leaf construction, recursive merges, and U-turn checks in a single
     NIF call. Used when depth >= `nif_depth_threshold` (default 2) and the NIF
     is loaded.
  2. **Cached** (`build_subtree_cached`) — pre-computes all `2^depth` leapfrog
     steps in one JIT call, then feeds them to the recursive Elixir `build_subtree`
     via an `:atomics`-counter closure. Used for depth >= 4 when NIF is unavailable.
  3. **Plain** (`build_subtree`) — calls `step_fn` per leaf. Fallback for shallow
     subtrees or when no `multi_step_fn` is provided.

  ## Speculative pre-computation

  When `multi_step_fn` is provided, the tree builder pre-computes entire
  forward and backward leapfrog chains from the initial position q0 in bulk,
  then slices into them for each subtree. This exploits the fact that all
  "go right" subtrees across doubling levels form one contiguous forward
  chain from q0, and all "go left" subtrees form one contiguous backward
  chain.

  The speculative buffer is lazily initialized on first need per direction
  and extended if deeper trees require more steps. This reduces JIT dispatch
  calls from O(max_depth) to O(1-2), saving ~250us per eliminated dispatch.

  Controlled by `Application.get_env(:exmc, :speculative_precompute, true)`.

  ## RNG

  Uses `:rand` for scalar random decisions (direction, proposal selection) for
  performance with BinaryBackend. The PRNG state is seeded deterministically
  from the caller's key.
  """

  alias Exmc.NUTS.{FaultInjector, NativeTree}

  require Logger

  @doc """
  Check if the Rust NIF tree builder is available.
  """
  def nif_available? do
    Code.ensure_loaded?(NativeTree) and function_exported?(NativeTree, :init_trajectory, 4)
  end

  @doc """
  Build a NUTS tree via iterative doubling.

  `rng` is an `:rand` state (seeded by caller for reproducibility).
  `multi_step_fn` is an optional batched leapfrog function from `BatchedLeapfrog.build/2`.
  When provided, subtrees are computed in a single JIT call for major speedup.

  Returns `%{q:, logp:, grad:, n_steps:, divergent:, accept_sum:, depth:}`.
  """
  def build(
        step_fn,
        q,
        p,
        logp,
        grad,
        epsilon,
        inv_mass_diag,
        max_depth,
        rng,
        joint_logp_0,
        multi_step_fn \\ nil,
        inv_mass_list \\ nil
      ) do
    joint_logp_0_scalar = Nx.to_number(joint_logp_0)

    # Full-tree NIF: all leapfrog steps pre-computed, entire tree built in
    # one Rust NIF call. Eliminates Elixir merge overhead (~200us/merge).
    # Gated on config + infrastructure availability.
    use_full_tree =
      Application.get_env(:exmc, :full_tree_nif, false) and
        multi_step_fn != nil and inv_mass_list != nil and
        nif_available?() and max_depth <= 10 and
        Process.get(:exmc_supervised, false) == false

    result =
      if use_full_tree do
        try do
          build_full_tree_nif(
            q,
            p,
            logp,
            grad,
            epsilon,
            inv_mass_diag,
            max_depth,
            rng,
            joint_logp_0_scalar,
            multi_step_fn
          )
        rescue
          e ->
            Logger.warning(
              "[NUTS.Tree] Full-tree NIF failed: #{Exception.message(e)}, falling back"
            )

            build_speculative(
              step_fn,
              q,
              p,
              logp,
              grad,
              epsilon,
              inv_mass_diag,
              max_depth,
              rng,
              joint_logp_0_scalar,
              multi_step_fn,
              inv_mass_list
            )
        end
      else
        build_speculative(
          step_fn,
          q,
          p,
          logp,
          grad,
          epsilon,
          inv_mass_diag,
          max_depth,
          rng,
          joint_logp_0_scalar,
          multi_step_fn,
          inv_mass_list
        )
      end

    # Track max observed depth for adaptive budget
    observed_max_depth = Process.get(:exmc_max_tree_depth_seen, 0)

    if result.depth > observed_max_depth do
      Process.put(:exmc_max_tree_depth_seen, result.depth)
    end

    result
  end

  # Full-tree NIF: pre-compute both forward and backward chains, then build
  # the entire NUTS tree in a single Rust NIF call.
  defp build_full_tree_nif(
         q,
         p,
         logp,
         grad,
         epsilon,
         inv_mass_diag,
         max_depth,
         rng,
         joint_logp_0,
         multi_step_fn
       ) do
    d = Nx.axis_size(q, 0)
    # Adaptive budget: use max observed depth + 1 headroom.
    # Rust bounds check terminates early if tree goes deeper than budget.
    # During early warmup (no data), use conservative 15 (depth 4).
    prev_max = Process.get(:exmc_max_tree_depth_seen, 0)

    budget_depth =
      if prev_max > 0 do
        min(prev_max + 1, max_depth)
      else
        min(max_depth, 4)
      end

    budget = min(trunc(:math.pow(2, budget_depth)) - 1, 1023)

    eps_t = Nx.tensor(epsilon, type: :f64, backend: Nx.BinaryBackend)
    neg_eps_t = Nx.tensor(-epsilon, type: :f64, backend: Nx.BinaryBackend)
    budget_t = Nx.tensor(budget, type: :s64)

    # Pre-compute forward chain (+epsilon) from q0
    {fwd_q, fwd_p, fwd_logp, fwd_grad} =
      multi_step_fn.(q, p, grad, eps_t, inv_mass_diag, budget_t)

    # Pre-compute backward chain (-epsilon) from q0
    {bwd_q, bwd_p, bwd_logp, bwd_grad} =
      multi_step_fn.(q, p, grad, neg_eps_t, inv_mass_diag, budget_t)

    # Backend copy to BinaryBackend for cheap binary extraction
    fwd_q = Nx.backend_copy(fwd_q, Nx.BinaryBackend)
    fwd_p = Nx.backend_copy(fwd_p, Nx.BinaryBackend)
    fwd_logp = Nx.backend_copy(fwd_logp, Nx.BinaryBackend)
    fwd_grad = Nx.backend_copy(fwd_grad, Nx.BinaryBackend)

    bwd_q = Nx.backend_copy(bwd_q, Nx.BinaryBackend)
    bwd_p = Nx.backend_copy(bwd_p, Nx.BinaryBackend)
    bwd_logp = Nx.backend_copy(bwd_logp, Nx.BinaryBackend)
    bwd_grad = Nx.backend_copy(bwd_grad, Nx.BinaryBackend)

    # Slice to valid rows and convert to binary
    fwd_q_bin = Nx.slice(fwd_q, [0, 0], [budget, d]) |> Nx.to_binary()
    fwd_p_bin = Nx.slice(fwd_p, [0, 0], [budget, d]) |> Nx.to_binary()
    fwd_logp_bin = Nx.slice(fwd_logp, [0], [budget]) |> Nx.to_binary()
    fwd_grad_bin = Nx.slice(fwd_grad, [0, 0], [budget, d]) |> Nx.to_binary()

    bwd_q_bin = Nx.slice(bwd_q, [0, 0], [budget, d]) |> Nx.to_binary()
    bwd_p_bin = Nx.slice(bwd_p, [0, 0], [budget, d]) |> Nx.to_binary()
    bwd_logp_bin = Nx.slice(bwd_logp, [0], [budget]) |> Nx.to_binary()
    bwd_grad_bin = Nx.slice(bwd_grad, [0, 0], [budget, d]) |> Nx.to_binary()

    # Initial state as binary
    q0_bin = Nx.to_binary(q)
    p0_bin = Nx.to_binary(p)
    grad0_bin = Nx.to_binary(grad)
    logp0 = Nx.to_number(logp)
    inv_mass_bin = Nx.to_binary(inv_mass_diag)

    # Seed Rust PRNG from Elixir PRNG
    {rng_seed_val, _rng} = :rand.uniform_s(rng)
    rng_seed = trunc(rng_seed_val * 1_000_000_000_000)

    # Guard joint_logp_0 against atoms
    jlp0 = if is_number(joint_logp_0), do: joint_logp_0, else: -1.0e300

    # Single NIF call: build entire tree
    nif_result =
      NativeTree.build_full_tree_bin(
        q0_bin,
        p0_bin,
        grad0_bin,
        logp0,
        fwd_q_bin,
        fwd_p_bin,
        fwd_logp_bin,
        fwd_grad_bin,
        bwd_q_bin,
        bwd_p_bin,
        bwd_logp_bin,
        bwd_grad_bin,
        inv_mass_bin,
        jlp0,
        max_depth,
        d,
        rng_seed
      )

    # Convert NIF result to standard return format
    %{
      q: Nx.from_binary(nif_result.q_bin, :f64, backend: Nx.BinaryBackend) |> Nx.reshape({d}),
      logp: Nx.tensor(nif_result.logp, type: :f64, backend: Nx.BinaryBackend),
      grad:
        Nx.from_binary(nif_result.grad_bin, :f64, backend: Nx.BinaryBackend) |> Nx.reshape({d}),
      n_steps: nif_result.n_steps,
      divergent: nif_result.divergent,
      accept_sum: nif_result.accept_sum,
      depth: nif_result.depth,
      recovered: false
    }
  end

  # Fallback path: speculative pre-computation + inner-subtree NIF/cached/plain dispatch.
  defp build_speculative(
         step_fn,
         q,
         p,
         logp,
         grad,
         epsilon,
         inv_mass_diag,
         max_depth,
         rng,
         joint_logp_0_scalar,
         multi_step_fn,
         inv_mass_list
       ) do
    # Pre-extract flat lists for U-turn checks (avoids Nx.to_flat_list per merge)
    q_list = Nx.to_flat_list(q)
    p_list = Nx.to_flat_list(p)

    initial = %{
      q_left: q,
      p_left: p,
      grad_left: grad,
      q_left_list: q_list,
      p_left_list: p_list,
      q_right: q,
      p_right: p,
      grad_right: grad,
      q_right_list: q_list,
      p_right_list: p_list,
      q_prop: q,
      logp_prop: logp,
      grad_prop: grad,
      rho_list: p_list,
      depth: 0,
      log_sum_weight: 0.0,
      n_steps: 0,
      divergent: false,
      accept_sum: 0.0,
      turning: false
    }

    # Speculative pre-computation buffer: pre-compute forward/backward leapfrog
    # chains from q0 in bulk, then slice into them for each subtree. This reduces
    # JIT dispatch calls from O(max_depth) to O(1-2).
    spec_buf =
      if multi_step_fn && Application.get_env(:exmc, :speculative_precompute, true) do
        %{
          fwd: nil,
          bwd: nil,
          initial_q: q,
          initial_p: p,
          initial_grad: grad,
          multi_step_fn: multi_step_fn,
          epsilon: epsilon,
          inv_mass_diag: inv_mass_diag,
          d: Nx.axis_size(q, 0)
        }
      end

    do_build(
      step_fn,
      initial,
      epsilon,
      inv_mass_diag,
      max_depth,
      rng,
      joint_logp_0_scalar,
      0,
      multi_step_fn,
      inv_mass_list,
      spec_buf
    )
  end

  defp do_build(
         _step_fn,
         traj,
         _epsilon,
         _inv_mass_diag,
         max_depth,
         _rng,
         _joint_logp_0,
         depth,
         _multi_step_fn,
         _inv_mass_list,
         _spec_buf
       )
       when depth >= max_depth do
    result(traj, depth)
  end

  defp do_build(
         _step_fn,
         %{divergent: true} = traj,
         _epsilon,
         _inv_mass_diag,
         _max_depth,
         _rng,
         _joint_logp_0,
         depth,
         _multi_step_fn,
         _inv_mass_list,
         _spec_buf
       ) do
    result(traj, depth)
  end

  defp do_build(
         _step_fn,
         %{turning: true} = traj,
         _epsilon,
         _inv_mass_diag,
         _max_depth,
         _rng,
         _joint_logp_0,
         depth,
         _multi_step_fn,
         _inv_mass_list,
         _spec_buf
       ) do
    result(traj, depth)
  end

  defp do_build(
         step_fn,
         traj,
         epsilon,
         inv_mass_diag,
         max_depth,
         rng,
         joint_logp_0,
         depth,
         multi_step_fn,
         inv_mass_list,
         spec_buf
       ) do
    # Random direction
    {rand_val, rng} = :rand.uniform_s(rng)
    go_right = rand_val > 0.5
    dir_epsilon = if go_right, do: epsilon, else: -epsilon

    # Build subtree from appropriate end (use endpoint gradient, not proposal gradient)
    {start_q, start_p, start_grad} =
      if go_right do
        {traj.q_right, traj.p_right, traj.grad_right}
      else
        {traj.q_left, traj.p_left, traj.grad_left}
      end

    supervised = Process.get(:exmc_supervised, false)
    n_steps = trunc(:math.pow(2, depth))
    direction = if go_right, do: :fwd, else: :bwd

    {subtree, rng, spec_buf} =
      if spec_buf do
        # Speculative path: ensure buffer has enough steps, slice, dispatch
        spec_buf = ensure_available(spec_buf, direction, n_steps)

        {sliced_q, sliced_p, sliced_logp, sliced_grad, spec_buf} =
          slice_precomputed(spec_buf, direction, n_steps)

        {sub, rng} =
          dispatch_subtree_precomputed(
            step_fn,
            start_q,
            start_p,
            start_grad,
            dir_epsilon,
            inv_mass_diag,
            depth,
            rng,
            joint_logp_0,
            inv_mass_list,
            sliced_q,
            sliced_p,
            sliced_logp,
            sliced_grad
          )

        {sub, rng, spec_buf}
      else
        # Non-speculative path (original)
        {sub, rng} =
          if supervised do
            safe_build_subtree(
              step_fn,
              start_q,
              start_p,
              start_grad,
              dir_epsilon,
              inv_mass_diag,
              depth,
              rng,
              joint_logp_0,
              multi_step_fn,
              inv_mass_list,
              supervised
            )
          else
            dispatch_subtree(
              step_fn,
              start_q,
              start_p,
              start_grad,
              dir_epsilon,
              inv_mass_diag,
              depth,
              rng,
              joint_logp_0,
              multi_step_fn,
              inv_mass_list
            )
          end

        {sub, rng, nil}
      end

    # Merge subtree into trajectory
    {new_traj, rng} =
      merge_trajectories(traj, subtree, go_right, inv_mass_diag, rng, inv_mass_list)

    do_build(
      step_fn,
      new_traj,
      epsilon,
      inv_mass_diag,
      max_depth,
      rng,
      joint_logp_0,
      depth + 1,
      multi_step_fn,
      inv_mass_list,
      spec_buf
    )
  end

  # --- Speculative pre-computation helpers ---
  # Pre-compute entire forward/backward leapfrog chains from q0 in bulk,
  # then slice into them for each subtree depth. Reduces JIT dispatch
  # calls from O(max_depth) to O(1-2).

  # Ensure the speculative buffer has at least `n_needed` steps available
  # beyond the current cursor for the given direction.
  defp ensure_available(%{} = spec_buf, direction, n_needed) do
    dir_buf = Map.get(spec_buf, direction)

    cond do
      # First time this direction is needed — compute initial batch
      is_nil(dir_buf) ->
        batch_size = max(32, n_needed * 2)
        dir_sign = if direction == :fwd, do: 1.0, else: -1.0

        eps_t =
          Nx.tensor(dir_sign * spec_buf.epsilon, type: :f64, backend: Nx.BinaryBackend)

        n_t = Nx.tensor(batch_size, type: :s64)

        {all_q, all_p, all_logp, all_grad} =
          spec_buf.multi_step_fn.(
            spec_buf.initial_q,
            spec_buf.initial_p,
            spec_buf.initial_grad,
            eps_t,
            spec_buf.inv_mass_diag,
            n_t
          )

        # Copy to BinaryBackend for cheap slicing
        all_q = Nx.backend_copy(all_q, Nx.BinaryBackend)
        all_p = Nx.backend_copy(all_p, Nx.BinaryBackend)
        all_logp = Nx.backend_copy(all_logp, Nx.BinaryBackend)
        all_grad = Nx.backend_copy(all_grad, Nx.BinaryBackend)

        d = spec_buf.d

        # Extract only the valid rows (batch_size out of max_steps)
        new_dir = %{
          all_q: Nx.slice(all_q, [0, 0], [batch_size, d]),
          all_p: Nx.slice(all_p, [0, 0], [batch_size, d]),
          all_logp: Nx.slice(all_logp, [0], [batch_size]),
          all_grad: Nx.slice(all_grad, [0, 0], [batch_size, d]),
          n_valid: batch_size,
          cursor: 0,
          last_q: Nx.slice(all_q, [batch_size - 1, 0], [1, d]) |> Nx.reshape({d}),
          last_p: Nx.slice(all_p, [batch_size - 1, 0], [1, d]) |> Nx.reshape({d}),
          last_grad: Nx.slice(all_grad, [batch_size - 1, 0], [1, d]) |> Nx.reshape({d})
        }

        Map.put(spec_buf, direction, new_dir)

      # Enough steps already available
      dir_buf.n_valid - dir_buf.cursor >= n_needed ->
        spec_buf

      # Need extension — compute more steps from where we left off
      true ->
        available = dir_buf.n_valid - dir_buf.cursor
        to_compute = max(32, n_needed - available + 16)
        dir_sign = if direction == :fwd, do: 1.0, else: -1.0

        eps_t =
          Nx.tensor(dir_sign * spec_buf.epsilon, type: :f64, backend: Nx.BinaryBackend)

        n_t = Nx.tensor(to_compute, type: :s64)

        {ext_q, ext_p, ext_logp, ext_grad} =
          spec_buf.multi_step_fn.(
            dir_buf.last_q,
            dir_buf.last_p,
            dir_buf.last_grad,
            eps_t,
            spec_buf.inv_mass_diag,
            n_t
          )

        ext_q = Nx.backend_copy(ext_q, Nx.BinaryBackend)
        ext_p = Nx.backend_copy(ext_p, Nx.BinaryBackend)
        ext_logp = Nx.backend_copy(ext_logp, Nx.BinaryBackend)
        ext_grad = Nx.backend_copy(ext_grad, Nx.BinaryBackend)

        d = spec_buf.d

        # Extract valid rows and concatenate with existing buffer
        ext_q_valid = Nx.slice(ext_q, [0, 0], [to_compute, d])
        ext_p_valid = Nx.slice(ext_p, [0, 0], [to_compute, d])
        ext_logp_valid = Nx.slice(ext_logp, [0], [to_compute])
        ext_grad_valid = Nx.slice(ext_grad, [0, 0], [to_compute, d])

        new_dir = %{
          dir_buf
          | all_q: Nx.concatenate([dir_buf.all_q, ext_q_valid]),
            all_p: Nx.concatenate([dir_buf.all_p, ext_p_valid]),
            all_logp: Nx.concatenate([dir_buf.all_logp, ext_logp_valid]),
            all_grad: Nx.concatenate([dir_buf.all_grad, ext_grad_valid]),
            n_valid: dir_buf.n_valid + to_compute,
            last_q: Nx.slice(ext_q, [to_compute - 1, 0], [1, d]) |> Nx.reshape({d}),
            last_p: Nx.slice(ext_p, [to_compute - 1, 0], [1, d]) |> Nx.reshape({d}),
            last_grad: Nx.slice(ext_grad, [to_compute - 1, 0], [1, d]) |> Nx.reshape({d})
        }

        Map.put(spec_buf, direction, new_dir)
    end
  end

  # Slice n_steps rows from the direction's buffer at the current cursor.
  defp slice_precomputed(%{} = spec_buf, direction, n_steps) do
    dir_buf = Map.get(spec_buf, direction)
    cursor = dir_buf.cursor
    d = spec_buf.d

    sub_q = Nx.slice(dir_buf.all_q, [cursor, 0], [n_steps, d])
    sub_p = Nx.slice(dir_buf.all_p, [cursor, 0], [n_steps, d])
    sub_logp = Nx.slice(dir_buf.all_logp, [cursor], [n_steps])
    sub_grad = Nx.slice(dir_buf.all_grad, [cursor, 0], [n_steps, d])

    # Advance cursor
    updated_dir = %{dir_buf | cursor: cursor + n_steps}
    spec_buf = Map.put(spec_buf, direction, updated_dir)

    {sub_q, sub_p, sub_logp, sub_grad, spec_buf}
  end

  # Dispatch pre-sliced tensors to NIF or cached path (no multi_step_fn call needed).
  defp dispatch_subtree_precomputed(
         step_fn,
         _start_q,
         start_p,
         start_grad,
         epsilon,
         inv_mass_diag,
         depth,
         rng,
         joint_logp_0,
         inv_mass_list,
         sliced_q,
         sliced_p,
         sliced_logp,
         sliced_grad
       ) do
    nif_threshold = Application.get_env(:exmc, :nif_depth_threshold, 2)

    use_nif =
      Application.get_env(:exmc, :use_nif, true) and nif_available?() and
        inv_mass_list != nil and depth >= nif_threshold

    if use_nif do
      build_subtree_nif_precomputed(
        sliced_q,
        sliced_p,
        sliced_logp,
        sliced_grad,
        inv_mass_diag,
        depth,
        rng,
        joint_logp_0
      )
    else
      build_subtree_cached_precomputed(
        sliced_q,
        sliced_p,
        sliced_logp,
        sliced_grad,
        start_p,
        start_grad,
        epsilon,
        inv_mass_diag,
        depth,
        rng,
        joint_logp_0,
        inv_mass_list,
        step_fn
      )
    end
  end

  # NIF path with pre-sliced tensors (already on BinaryBackend).
  # Skips multi_step_fn call — tensors come from speculative buffer.
  defp build_subtree_nif_precomputed(
         sliced_q,
         sliced_p,
         sliced_logp,
         sliced_grad,
         inv_mass_diag,
         depth,
         rng,
         joint_logp_0
       ) do
    n_steps = trunc(:math.pow(2, depth))
    d = Nx.axis_size(sliced_q, 1)

    # Convert to binaries (already BinaryBackend — cheap)
    all_q_bin = Nx.slice(sliced_q, [0, 0], [n_steps, d]) |> Nx.to_binary()
    all_p_bin = Nx.slice(sliced_p, [0, 0], [n_steps, d]) |> Nx.to_binary()
    all_logp_bin = Nx.slice(sliced_logp, [0], [n_steps]) |> Nx.to_binary()
    all_grad_bin = Nx.slice(sliced_grad, [0, 0], [n_steps, d]) |> Nx.to_binary()

    inv_mass_bin = Nx.to_binary(inv_mass_diag)

    # Seed Rust PRNG from Elixir PRNG
    {rng_seed_val, rng} = :rand.uniform_s(rng)
    rng_seed = trunc(rng_seed_val * 1_000_000_000_000)

    jlp0 = if is_number(joint_logp_0), do: joint_logp_0, else: -1.0e300

    # Backward chain states were computed with -epsilon, so the NIF traverses
    # them as if going forward (going_right=true always).
    nif_result =
      NativeTree.build_subtree_bin(
        all_q_bin,
        all_p_bin,
        all_logp_bin,
        all_grad_bin,
        inv_mass_bin,
        jlp0,
        depth,
        d,
        true,
        rng_seed
      )

    subtree = nif_subtree_to_elixir(nif_result, d)
    {subtree, rng}
  end

  # Cached path with pre-sliced tensors (already on BinaryBackend).
  # Creates a cached_step_fn from the sliced tensors, then runs the
  # same recursive build_subtree for correct RNG consumption.
  defp build_subtree_cached_precomputed(
         sliced_q,
         sliced_p,
         sliced_logp,
         sliced_grad,
         start_p,
         start_grad,
         epsilon,
         inv_mass_diag,
         depth,
         rng,
         joint_logp_0,
         inv_mass_list,
         _step_fn
       ) do
    n_steps = trunc(:math.pow(2, depth))
    d = Nx.axis_size(sliced_q, 1)

    raw_logps = Nx.to_flat_list(Nx.slice(sliced_logp, [0], [n_steps])) |> List.to_tuple()

    counter = :atomics.new(1, signed: false)
    half = Nx.tensor(0.5, type: :f64, backend: Nx.BinaryBackend)

    cached_step_fn = fn _q, _p, _grad, _eps, _inv_mass ->
      idx = :atomics.get(counter, 1)
      :atomics.put(counter, 1, idx + 1)

      q_new = Nx.slice(sliced_q, [idx, 0], [1, d]) |> Nx.reshape({d})
      p_new = Nx.slice(sliced_p, [idx, 0], [1, d]) |> Nx.reshape({d})
      logp_new = Nx.tensor(elem(raw_logps, idx), type: :f64, backend: Nx.BinaryBackend)
      grad_new = Nx.slice(sliced_grad, [idx, 0], [1, d]) |> Nx.reshape({d})

      ke = Nx.multiply(half, Nx.sum(Nx.multiply(p_new, Nx.multiply(inv_mass_diag, p_new))))
      jlp = Nx.subtract(logp_new, ke)

      {q_new, p_new, logp_new, grad_new, jlp}
    end

    # The cached_step_fn ignores its q/p/grad inputs — it reads from the
    # pre-sliced buffer by atomic counter. We pass dummy q; p and grad are
    # used only for the initial subtree structure (not passed to step_fn at
    # depth 0 — step_fn is called, but it ignores q/p/grad args).
    dummy_q = Nx.broadcast(Nx.tensor(0.0, type: :f64, backend: Nx.BinaryBackend), {d})

    build_subtree(
      cached_step_fn,
      dummy_q,
      start_p,
      start_grad,
      epsilon,
      inv_mass_diag,
      depth,
      rng,
      joint_logp_0,
      inv_mass_list
    )
  end

  # Dispatch to the appropriate subtree builder (NIF, cached, or plain).
  defp dispatch_subtree(
         step_fn,
         q,
         p,
         grad,
         epsilon,
         inv_mass_diag,
         depth,
         rng,
         joint_logp_0,
         multi_step_fn,
         inv_mass_list
       ) do
    nif_threshold = Application.get_env(:exmc, :nif_depth_threshold, 2)

    use_nif_subtree =
      Application.get_env(:exmc, :use_nif, true) and nif_available?() and
        multi_step_fn != nil and inv_mass_list != nil and depth >= nif_threshold

    cond do
      use_nif_subtree ->
        build_subtree_nif(
          multi_step_fn,
          q,
          p,
          grad,
          epsilon,
          inv_mass_diag,
          depth,
          rng,
          joint_logp_0,
          inv_mass_list
        )

      multi_step_fn && depth >= 4 ->
        build_subtree_cached(
          multi_step_fn,
          q,
          p,
          grad,
          epsilon,
          inv_mass_diag,
          depth,
          rng,
          joint_logp_0,
          inv_mass_list
        )

      true ->
        build_subtree(
          step_fn,
          q,
          p,
          grad,
          epsilon,
          inv_mass_diag,
          depth,
          rng,
          joint_logp_0,
          inv_mass_list
        )
    end
  end

  # Fault-tolerant subtree wrapper. On crash, returns a divergent placeholder.
  defp safe_build_subtree(
         step_fn,
         q,
         p,
         grad,
         epsilon,
         inv_mass_diag,
         depth,
         rng,
         joint_logp_0,
         multi_step_fn,
         inv_mass_list,
         supervised
       ) do
    case supervised do
      true ->
        try do
          dispatch_subtree(
            step_fn,
            q,
            p,
            grad,
            epsilon,
            inv_mass_diag,
            depth,
            rng,
            joint_logp_0,
            multi_step_fn,
            inv_mass_list
          )
        rescue
          e ->
            Logger.warning("[NUTS.Tree] Subtree crash at depth #{depth}: #{Exception.message(e)}")
            divergent_placeholder(q, p, grad, depth)
        end

      :task ->
        timeout = Process.get(:exmc_supervised_timeout, 30_000)

        task =
          Task.async(fn ->
            dispatch_subtree(
              step_fn,
              q,
              p,
              grad,
              epsilon,
              inv_mass_diag,
              depth,
              rng,
              joint_logp_0,
              multi_step_fn,
              inv_mass_list
            )
          end)

        case Task.yield(task, timeout) || Task.shutdown(task) do
          {:ok, result} ->
            result

          nil ->
            Logger.warning("[NUTS.Tree] Subtree timed out at depth #{depth} (#{timeout}ms)")
            divergent_placeholder(q, p, grad, depth)

          {:exit, reason} ->
            Logger.warning(
              "[NUTS.Tree] Subtree task exited at depth #{depth}: #{inspect(reason)}"
            )

            divergent_placeholder(q, p, grad, depth)
        end

      _ ->
        dispatch_subtree(
          step_fn,
          q,
          p,
          grad,
          epsilon,
          inv_mass_diag,
          depth,
          rng,
          joint_logp_0,
          multi_step_fn,
          inv_mass_list
        )
    end
  end

  # Base case: single leapfrog step
  defp build_subtree(
         step_fn,
         q,
         p,
         grad,
         epsilon,
         inv_mass_diag,
         0,
         rng,
         joint_logp_0,
         _inv_mass_list
       ) do
    FaultInjector.maybe_fault!(0)

    {q_new, p_new, logp_new, grad_new, joint_logp_t} =
      step_fn.(q, p, grad, epsilon, inv_mass_diag)

    # Copy EXLA.Backend outputs to BinaryBackend for cheap Elixir-side arithmetic.
    q_new = Nx.backend_copy(q_new, Nx.BinaryBackend)
    p_new = Nx.backend_copy(p_new, Nx.BinaryBackend)
    logp_new = Nx.backend_copy(logp_new, Nx.BinaryBackend)
    grad_new = Nx.backend_copy(grad_new, Nx.BinaryBackend)

    # joint_logp computed inside JIT — just extract the scalar
    joint_logp_new = Nx.to_number(joint_logp_t)

    # Guard against NaN/Inf from numerical issues (e.g., log(0) gradient)
    {divergent, log_weight, accept_prob} =
      if is_number(joint_logp_new) do
        d = joint_logp_new - joint_logp_0
        {d < -1000.0, d, min(1.0, :math.exp(min(d, 0.0)))}
      else
        {true, -1001.0, 0.0}
      end

    # Pre-extract flat lists once at leaf (avoids repeated Nx.to_flat_list in merges)
    q_list = Nx.to_flat_list(q_new)
    p_list = Nx.to_flat_list(p_new)

    subtree = %{
      q_left: q_new,
      p_left: p_new,
      grad_left: grad_new,
      q_left_list: q_list,
      p_left_list: p_list,
      q_right: q_new,
      p_right: p_new,
      grad_right: grad_new,
      q_right_list: q_list,
      p_right_list: p_list,
      q_prop: q_new,
      logp_prop: logp_new,
      grad_prop: grad_new,
      rho_list: p_list,
      depth: 0,
      log_sum_weight: log_weight,
      n_steps: 1,
      divergent: divergent,
      accept_sum: accept_prob,
      turning: false
    }

    {subtree, rng}
  end

  # Recursive case: build two half-subtrees and merge
  defp build_subtree(
         step_fn,
         q,
         p,
         grad,
         epsilon,
         inv_mass_diag,
         depth,
         rng,
         joint_logp_0,
         inv_mass_list
       )
       when depth > 0 do
    FaultInjector.maybe_fault!(depth)
    half_depth = depth - 1

    # Build first half
    {first, rng} =
      build_subtree(
        step_fn,
        q,
        p,
        grad,
        epsilon,
        inv_mass_diag,
        half_depth,
        rng,
        joint_logp_0,
        inv_mass_list
      )

    if first.divergent or first.turning do
      {first, rng}
    else
      # Build second half from appropriate end (use endpoint gradient)
      {next_q, next_p, next_grad} =
        if epsilon > 0 do
          {first.q_right, first.p_right, first.grad_right}
        else
          {first.q_left, first.p_left, first.grad_left}
        end

      {second, rng} =
        build_subtree(
          step_fn,
          next_q,
          next_p,
          next_grad,
          epsilon,
          inv_mass_diag,
          half_depth,
          rng,
          joint_logp_0,
          inv_mass_list
        )

      {merged, rng} = merge_subtrees(first, second, epsilon, inv_mass_diag, rng, inv_mass_list)
      {merged, rng}
    end
  end

  # --- ETS-cached batched subtree builder ---
  # Pre-computes all 2^depth leapfrog steps in one JIT call (eliminating
  # per-step dispatch overhead), then stores results in an ETS table.
  # The existing recursive build_subtree runs unchanged with a cached
  # step_fn that reads from ETS by counter. This guarantees identical
  # RNG consumption, early termination, and merge behavior.
  #
  # Trade-off: loses early termination within the subtree. For depth 8,
  # worst case wastes 128 extra steps. At ~3us/step inside JIT, that's
  # only ~0.4ms wasted vs the ~32ms dispatch overhead saved.

  defp build_subtree_cached(
         multi_step_fn,
         q,
         p,
         grad,
         epsilon,
         inv_mass_diag,
         depth,
         rng,
         joint_logp_0,
         inv_mass_list
       ) do
    n_steps = trunc(:math.pow(2, depth))
    d = Nx.axis_size(q, 0)

    # Pre-compute all leapfrog steps in one JIT call
    eps_t = Nx.tensor(epsilon, type: :f64, backend: Nx.BinaryBackend)
    n_t = Nx.tensor(n_steps, type: :s64)
    {all_q, all_p, all_logp, all_grad} = multi_step_fn.(q, p, grad, eps_t, inv_mass_diag, n_t)

    # Copy all results to BinaryBackend (4 bulk tensors vs 4*n_steps individual copies)
    all_q = Nx.backend_copy(all_q, Nx.BinaryBackend)
    all_p = Nx.backend_copy(all_p, Nx.BinaryBackend)
    all_logp = Nx.backend_copy(all_logp, Nx.BinaryBackend)
    all_grad = Nx.backend_copy(all_grad, Nx.BinaryBackend)
    raw_logps = Nx.to_flat_list(Nx.slice(all_logp, [0], [n_steps])) |> List.to_tuple()

    # Atomic counter — the cached_step_fn slices lazily on each call,
    # so only steps actually visited by the tree builder are extracted.
    counter = :atomics.new(1, signed: false)

    # Cached step_fn: slices the i-th row from pre-computed tensors.
    # Returns the same 5-tuple as the real step_fn.
    half = Nx.tensor(0.5, type: :f64, backend: Nx.BinaryBackend)

    cached_step_fn = fn _q, _p, _grad, _eps, _inv_mass ->
      idx = :atomics.get(counter, 1)
      :atomics.put(counter, 1, idx + 1)

      q_new = Nx.slice(all_q, [idx, 0], [1, d]) |> Nx.reshape({d})
      p_new = Nx.slice(all_p, [idx, 0], [1, d]) |> Nx.reshape({d})
      logp_new = Nx.tensor(elem(raw_logps, idx), type: :f64, backend: Nx.BinaryBackend)
      grad_new = Nx.slice(all_grad, [idx, 0], [1, d]) |> Nx.reshape({d})

      ke = Nx.multiply(half, Nx.sum(Nx.multiply(p_new, Nx.multiply(inv_mass_diag, p_new))))
      jlp = Nx.subtract(logp_new, ke)

      {q_new, p_new, logp_new, grad_new, jlp}
    end

    # Run the same recursive build_subtree — identical RNG, early termination, merges
    build_subtree(
      cached_step_fn,
      q,
      p,
      grad,
      epsilon,
      inv_mass_diag,
      depth,
      rng,
      joint_logp_0,
      inv_mass_list
    )
  end

  # --- NIF-accelerated subtree builder ---
  # Replaces build_subtree_cached's inner logic: instead of per-leaf Nx.slice +
  # Elixir recursive merges, sends all pre-computed states as binaries to Rust
  # which builds the entire subtree (leaves + merges + U-turn checks) in one call.
  #
  # The outer loop and top-level merges stay in Elixir (only ~log(max_depth) merges).

  defp build_subtree_nif(
         multi_step_fn,
         q,
         p,
         grad,
         epsilon,
         inv_mass_diag,
         depth,
         rng,
         joint_logp_0,
         _inv_mass_list
       ) do
    n_steps = trunc(:math.pow(2, depth))
    d = Nx.axis_size(q, 0)
    going_right = epsilon > 0

    # Pre-compute all leapfrog steps in one JIT call (same as build_subtree_cached)
    eps_t = Nx.tensor(epsilon, type: :f64, backend: Nx.BinaryBackend)
    n_t = Nx.tensor(n_steps, type: :s64)
    {all_q, all_p, all_logp, all_grad} = multi_step_fn.(q, p, grad, eps_t, inv_mass_diag, n_t)

    # Backend copy to BinaryBackend, then extract binaries
    all_q_b = Nx.backend_copy(all_q, Nx.BinaryBackend)
    all_p_b = Nx.backend_copy(all_p, Nx.BinaryBackend)
    all_logp_b = Nx.backend_copy(all_logp, Nx.BinaryBackend)
    all_grad_b = Nx.backend_copy(all_grad, Nx.BinaryBackend)

    all_q_bin = Nx.slice(all_q_b, [0, 0], [n_steps, d]) |> Nx.to_binary()
    all_p_bin = Nx.slice(all_p_b, [0, 0], [n_steps, d]) |> Nx.to_binary()
    all_logp_bin = Nx.slice(all_logp_b, [0], [n_steps]) |> Nx.to_binary()
    all_grad_bin = Nx.slice(all_grad_b, [0, 0], [n_steps, d]) |> Nx.to_binary()

    inv_mass_bin = Nx.to_binary(inv_mass_diag)

    # Seed Rust PRNG from Elixir PRNG
    {rng_seed_val, rng} = :rand.uniform_s(rng)
    rng_seed = trunc(rng_seed_val * 1_000_000_000_000)

    # Guard joint_logp_0 against atoms
    jlp0 = if is_number(joint_logp_0), do: joint_logp_0, else: -1.0e300

    # NIF: build entire subtree in Rust
    nif_result =
      NativeTree.build_subtree_bin(
        all_q_bin,
        all_p_bin,
        all_logp_bin,
        all_grad_bin,
        inv_mass_bin,
        jlp0,
        depth,
        d,
        going_right,
        rng_seed
      )

    # Convert NIF result (binary endpoints) back to Elixir subtree map
    subtree = nif_subtree_to_elixir(nif_result, d)
    {subtree, rng}
  end

  # Convert NIF subtree map (with binary data) to Elixir tree map (with Nx tensors + lists)
  defp nif_subtree_to_elixir(r, d) do
    q_left = Nx.from_binary(r.q_left_bin, :f64, backend: Nx.BinaryBackend) |> Nx.reshape({d})
    p_left = Nx.from_binary(r.p_left_bin, :f64, backend: Nx.BinaryBackend) |> Nx.reshape({d})
    q_right = Nx.from_binary(r.q_right_bin, :f64, backend: Nx.BinaryBackend) |> Nx.reshape({d})
    p_right = Nx.from_binary(r.p_right_bin, :f64, backend: Nx.BinaryBackend) |> Nx.reshape({d})

    # Decode rho from NIF binary
    rho_list =
      r.rho_bin
      |> then(&Nx.from_binary(&1, :f64, backend: Nx.BinaryBackend))
      |> Nx.reshape({d})
      |> Nx.to_flat_list()

    %{
      q_left: q_left,
      p_left: p_left,
      grad_left:
        Nx.from_binary(r.grad_left_bin, :f64, backend: Nx.BinaryBackend) |> Nx.reshape({d}),
      q_left_list: Nx.to_flat_list(q_left),
      p_left_list: Nx.to_flat_list(p_left),
      q_right: q_right,
      p_right: p_right,
      grad_right:
        Nx.from_binary(r.grad_right_bin, :f64, backend: Nx.BinaryBackend) |> Nx.reshape({d}),
      q_right_list: Nx.to_flat_list(q_right),
      p_right_list: Nx.to_flat_list(p_right),
      q_prop: Nx.from_binary(r.q_prop_bin, :f64, backend: Nx.BinaryBackend) |> Nx.reshape({d}),
      logp_prop: Nx.tensor(r.logp_prop, type: :f64, backend: Nx.BinaryBackend),
      grad_prop:
        Nx.from_binary(r.grad_prop_bin, :f64, backend: Nx.BinaryBackend) |> Nx.reshape({d}),
      rho_list: rho_list,
      log_sum_weight: r.log_sum_weight,
      n_steps: r.n_steps,
      divergent: r.divergent,
      accept_sum: r.accept_sum,
      turning: r.turning,
      depth: r.depth
    }
  end

  # --- Merge operations ---

  defp merge_subtrees(first, second, epsilon, inv_mass_diag, rng, inv_mass_list) do
    combined_log_weight = log_sum_exp(first.log_sum_weight, second.log_sum_weight)
    combined_n_steps = first.n_steps + second.n_steps
    combined_accept_sum = first.accept_sum + second.accept_sum
    combined_divergent = first.divergent or second.divergent

    # Multinomial: accept second's proposal with prob exp(second.log_weight - combined_log_weight)
    {rand_val, rng} = :rand.uniform_s(rng)
    accept_prob = :math.exp(second.log_sum_weight - combined_log_weight)
    use_second = rand_val < accept_prob

    {q_prop, logp_prop, grad_prop} =
      if use_second do
        {second.q_prop, second.logp_prop, second.grad_prop}
      else
        {first.q_prop, first.logp_prop, first.grad_prop}
      end

    # Cumulative momentum sum: ρ = ρ_first + ρ_second
    rho_list = zip_add(first.rho_list, second.rho_list)

    # Endpoints depend on direction (propagate pre-extracted lists)
    {q_left, p_left, grad_left, q_left_list, p_left_list, q_right, p_right, grad_right,
     q_right_list,
     p_right_list} =
      if epsilon > 0 do
        {first.q_left, first.p_left, first.grad_left, first.q_left_list, first.p_left_list,
         second.q_right, second.p_right, second.grad_right, second.q_right_list,
         second.p_right_list}
      else
        {second.q_left, second.p_left, second.grad_left, second.q_left_list, second.p_left_list,
         first.q_right, first.p_right, first.grad_right, first.q_right_list, first.p_right_list}
      end

    # U-turn checks (Betancourt 2017 generalized criterion: ρ · (M^{-1} p±) < 0)
    iml = inv_mass_list || Nx.to_flat_list(inv_mass_diag)

    # Check 1: Full trajectory U-turn
    turning =
      combined_divergent or second.turning or
        check_uturn_rho(rho_list, p_left_list, p_right_list, iml)

    # Checks 2 & 3: Sub-trajectory U-turn (only when children have > 1 leaf)
    turning =
      if not turning and first.depth > 0 do
        {left_sub, right_sub} = if epsilon > 0, do: {first, second}, else: {second, first}

        # Check 2: left sub-trajectory + first point of right sub-trajectory
        partial_rho_2 = zip_add(left_sub.rho_list, right_sub.p_left_list)

        if check_uturn_rho(partial_rho_2, left_sub.p_left_list, right_sub.p_left_list, iml) do
          true
        else
          # Check 3: last point of left sub-trajectory + right sub-trajectory
          partial_rho_3 = zip_add(left_sub.p_right_list, right_sub.rho_list)
          check_uturn_rho(partial_rho_3, left_sub.p_right_list, right_sub.p_right_list, iml)
        end
      else
        turning
      end

    merged = %{
      q_left: q_left,
      p_left: p_left,
      grad_left: grad_left,
      q_left_list: q_left_list,
      p_left_list: p_left_list,
      q_right: q_right,
      p_right: p_right,
      grad_right: grad_right,
      q_right_list: q_right_list,
      p_right_list: p_right_list,
      q_prop: q_prop,
      logp_prop: logp_prop,
      grad_prop: grad_prop,
      rho_list: rho_list,
      depth: max(first.depth, second.depth) + 1,
      log_sum_weight: combined_log_weight,
      n_steps: combined_n_steps,
      divergent: combined_divergent,
      accept_sum: combined_accept_sum,
      turning: turning,
      recovered: Map.get(first, :recovered, false) or Map.get(second, :recovered, false)
    }

    {merged, rng}
  end

  defp merge_trajectories(traj, subtree, go_right, inv_mass_diag, rng, inv_mass_list) do
    combined_log_weight = log_sum_exp(traj.log_sum_weight, subtree.log_sum_weight)
    combined_n_steps = traj.n_steps + subtree.n_steps
    combined_accept_sum = traj.accept_sum + subtree.accept_sum
    combined_divergent = traj.divergent or subtree.divergent

    # Biased progressive sampling (Stan/PyMC): accept subtree with prob
    # min(1, exp(subtree.lsw - traj.lsw)). Uses OLD trajectory weight, not combined.
    # When subtree outweighs trajectory, always accept. This reduces "sticky" q_0
    # selection and improves ESS. See Betancourt 2017 Appendix A.3.2.
    {rand_val, rng} = :rand.uniform_s(rng)
    use_subtree = :math.log(rand_val) < (subtree.log_sum_weight - traj.log_sum_weight)

    {q_prop, logp_prop, grad_prop} =
      if use_subtree do
        {subtree.q_prop, subtree.logp_prop, subtree.grad_prop}
      else
        {traj.q_prop, traj.logp_prop, traj.grad_prop}
      end

    # Cumulative momentum sum: ρ = ρ_traj + ρ_subtree
    rho_list = zip_add(traj.rho_list, subtree.rho_list)

    # Update endpoints (propagate pre-extracted lists)
    {q_left, p_left, grad_left, q_left_list, p_left_list, q_right, p_right, grad_right,
     q_right_list,
     p_right_list} =
      if go_right do
        {traj.q_left, traj.p_left, traj.grad_left, traj.q_left_list, traj.p_left_list,
         subtree.q_right, subtree.p_right, subtree.grad_right, subtree.q_right_list,
         subtree.p_right_list}
      else
        {subtree.q_left, subtree.p_left, subtree.grad_left, subtree.q_left_list,
         subtree.p_left_list, traj.q_right, traj.p_right, traj.grad_right, traj.q_right_list,
         traj.p_right_list}
      end

    # U-turn checks (Betancourt 2017 generalized criterion: ρ · (M^{-1} p±) < 0)
    iml = inv_mass_list || Nx.to_flat_list(inv_mass_diag)

    # Check 1: Full trajectory U-turn
    turning =
      combined_divergent or subtree.turning or
        check_uturn_rho(rho_list, p_left_list, p_right_list, iml)

    # Checks 2 & 3: Sub-trajectory U-turn
    turning =
      if not turning do
        {left_sub, right_sub} = if go_right, do: {traj, subtree}, else: {subtree, traj}

        # Check 2: left sub-trajectory + first point of right sub-trajectory
        partial_rho_2 = zip_add(left_sub.rho_list, right_sub.p_left_list)

        if check_uturn_rho(partial_rho_2, left_sub.p_left_list, right_sub.p_left_list, iml) do
          true
        else
          # Check 3: last point of left sub-trajectory + right sub-trajectory
          partial_rho_3 = zip_add(left_sub.p_right_list, right_sub.rho_list)
          check_uturn_rho(partial_rho_3, left_sub.p_right_list, right_sub.p_right_list, iml)
        end
      else
        turning
      end

    merged = %{
      q_left: q_left,
      p_left: p_left,
      grad_left: grad_left,
      q_left_list: q_left_list,
      p_left_list: p_left_list,
      q_right: q_right,
      p_right: p_right,
      grad_right: grad_right,
      q_right_list: q_right_list,
      p_right_list: p_right_list,
      q_prop: q_prop,
      logp_prop: logp_prop,
      grad_prop: grad_prop,
      rho_list: rho_list,
      depth: traj.depth + 1,
      log_sum_weight: combined_log_weight,
      n_steps: combined_n_steps,
      divergent: combined_divergent,
      accept_sum: combined_accept_sum,
      turning: turning,
      recovered: Map.get(traj, :recovered, false) or Map.get(subtree, :recovered, false)
    }

    {merged, rng}
  end

  # --- U-turn checks ---

  # Generalized U-turn check (Betancourt 2017) using cumulative momentum sum ρ.
  # Checks: ρ · (M^{-1} p_left) < 0 OR ρ · (M^{-1} p_right) < 0
  # Unlike the endpoint criterion (q+-q-), this weights all components uniformly
  # regardless of inv_mass scale, preventing premature termination in hierarchical
  # models with large inv_mass range.
  defp check_uturn_rho(rho, pl, pr, inv_mass_list) do
    {dot_right, dot_left} = zip_reduce_rho(rho, pl, pr, inv_mass_list, 0.0, 0.0)
    dot_right < 0.0 or dot_left < 0.0
  end

  defp zip_reduce_rho([], [], [], [], dr, dl), do: {dr, dl}

  defp zip_reduce_rho([r | rs], [pl | pls], [pr | prs], [im | ims], dr, dl) do
    v = r * im
    zip_reduce_rho(rs, pls, prs, ims, dr + v * pr, dl + v * pl)
  end

  # Element-wise list addition (for accumulating momentum sum ρ)
  defp zip_add([], []), do: []
  defp zip_add([a | as_], [b | bs]), do: [a + b | zip_add(as_, bs)]

  # --- Helpers ---

  defp log_sum_exp(a, b) do
    m = max(a, b)

    if m == :neg_infinity or m == -1.0e300 do
      -1.0e300
    else
      m + :math.log(:math.exp(a - m) + :math.exp(b - m))
    end
  end

  defp result(traj, depth) do
    %{
      q: traj.q_prop,
      logp: traj.logp_prop,
      grad: traj.grad_prop,
      n_steps: traj.n_steps,
      divergent: traj.divergent,
      accept_sum: traj.accept_sum,
      depth: depth,
      recovered: Map.get(traj, :recovered, false)
    }
  end

  # Build a divergent placeholder subtree from the starting state.
  # Used when a subtree crashes and must be replaced with a valid structure.
  # Uses a fresh RNG (post-crash trajectories are valid but not deterministic).
  defp divergent_placeholder(q, p, grad, depth) do
    q_list = Nx.to_flat_list(q)
    p_list = Nx.to_flat_list(p)
    n_steps = trunc(:math.pow(2, depth))
    recovery_rng = :rand.seed_s(:exsss, System.unique_integer([:positive]))

    subtree = %{
      q_left: q,
      p_left: p,
      grad_left: grad,
      q_left_list: q_list,
      p_left_list: p_list,
      q_right: q,
      p_right: p,
      grad_right: grad,
      q_right_list: q_list,
      p_right_list: p_list,
      q_prop: q,
      logp_prop: Nx.tensor(-1.0e300, type: :f64, backend: Nx.BinaryBackend),
      grad_prop: grad,
      rho_list: p_list,
      depth: depth,
      log_sum_weight: -1001.0,
      n_steps: n_steps,
      divergent: true,
      accept_sum: 0.0,
      turning: false,
      recovered: true
    }

    {subtree, recovery_rng}
  end
end
