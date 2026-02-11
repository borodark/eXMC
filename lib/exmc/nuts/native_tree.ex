defmodule Exmc.NUTS.NativeTree do
  @moduledoc """
  Rust NIF bindings for the NUTS tree builder.

  Provides both list-based (for tests) and binary-based (for performance) APIs.
  The binary API uses `Nx.to_binary()`/`Nx.from_binary()` to avoid Erlang
  term encoding overhead for large tensor data.
  """

  use Rustler, otp_app: :exmc, crate: :exmc_tree

  # --- Binary API (performance path) ---

  @doc "Create a new trajectory from binary position data."
  def init_trajectory_bin(_q_bin, _p_bin, _grad_bin, _logp),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc "Get endpoint as raw binaries. Returns {q_bin, p_bin, grad_bin}."
  def get_endpoint_bin(_traj_ref, _go_right), do: :erlang.nif_error(:nif_not_loaded)

  @doc "Build subtree from binary pre-computed states and merge into trajectory."
  def build_and_merge_bin(
        _traj_ref,
        _all_q_bin,
        _all_p_bin,
        _all_logp_bin,
        _all_grad_bin,
        _inv_mass_bin,
        _joint_logp_0,
        _depth,
        _d,
        _go_right,
        _rng_seed
      ),
      do: :erlang.nif_error(:nif_not_loaded)

  @doc "Build subtree from binary pre-computed states. Returns full subtree as map."
  def build_subtree_bin(
        _all_q_bin,
        _all_p_bin,
        _all_logp_bin,
        _all_grad_bin,
        _inv_mass_bin,
        _joint_logp_0,
        _depth,
        _d,
        _going_right,
        _rng_seed
      ),
      do: :erlang.nif_error(:nif_not_loaded)

  @doc "Build full NUTS tree from pre-computed forward/backward chains. Returns result map."
  def build_full_tree_bin(
        _q0_bin,
        _p0_bin,
        _grad0_bin,
        _logp0,
        _fwd_q_bin,
        _fwd_p_bin,
        _fwd_logp_bin,
        _fwd_grad_bin,
        _bwd_q_bin,
        _bwd_p_bin,
        _bwd_logp_bin,
        _bwd_grad_bin,
        _inv_mass_bin,
        _joint_logp_0,
        _max_depth,
        _d,
        _rng_seed
      ),
      do: :erlang.nif_error(:nif_not_loaded)

  @doc "Extract final result with binary q/grad. Returns map with :q_bin, :grad_bin."
  def get_result_bin(_traj_ref), do: :erlang.nif_error(:nif_not_loaded)

  # --- List API (test convenience) ---

  @doc "Create a new trajectory from lists."
  def init_trajectory(_q_list, _p_list, _grad_list, _logp), do: :erlang.nif_error(:nif_not_loaded)

  @doc "Check if trajectory has terminated."
  def is_terminated(_traj_ref), do: :erlang.nif_error(:nif_not_loaded)

  @doc "Get endpoint as lists. Returns {q_list, p_list, grad_list}."
  def get_endpoint(_traj_ref, _go_right), do: :erlang.nif_error(:nif_not_loaded)

  @doc "Build subtree from list pre-computed states and merge into trajectory."
  def build_and_merge(
        _traj_ref,
        _all_q,
        _all_p,
        _all_logp,
        _all_grad,
        _inv_mass,
        _joint_logp_0,
        _depth,
        _d,
        _go_right,
        _rng_seed
      ),
      do: :erlang.nif_error(:nif_not_loaded)

  @doc "Extract final result with list q/grad."
  def get_result(_traj_ref), do: :erlang.nif_error(:nif_not_loaded)
end
