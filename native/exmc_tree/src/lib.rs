mod math;
mod tree;
mod types;
mod uturn;

use std::sync::Mutex;

use rand::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;
use rustler::{Atom, Binary, Encoder, Env, NewBinary, NifResult, ResourceArc, Term};

use types::{PrecomputedStates, Trajectory};

/// Wrapper for Trajectory behind a Mutex (ResourceArc requires Send + Sync).
pub struct TrajectoryResource(Mutex<Trajectory>);

// --- Helper: convert bytes to Vec<f64> ---

fn bytes_to_f64_vec(bytes: &[u8]) -> Vec<f64> {
    bytes
        .chunks_exact(8)
        .map(|chunk| f64::from_ne_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn f64_slice_to_binary<'a>(env: Env<'a>, data: &[f64]) -> Binary<'a> {
    let byte_len = data.len() * 8;
    let mut bin = NewBinary::new(env, byte_len);
    let src = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, byte_len) };
    bin.as_mut_slice().copy_from_slice(src);
    bin.into()
}

/// Initialize a trajectory from binary position data.
///
/// All inputs are raw f64 binaries (from `Nx.to_binary()`).
#[rustler::nif(schedule = "DirtyCpu")]
fn init_trajectory_bin(
    q_bin: Binary,
    p_bin: Binary,
    grad_bin: Binary,
    logp: f64,
) -> ResourceArc<TrajectoryResource> {
    let q = bytes_to_f64_vec(q_bin.as_slice());
    let p = bytes_to_f64_vec(p_bin.as_slice());
    let grad = bytes_to_f64_vec(grad_bin.as_slice());
    let traj = Trajectory::new(q, p, grad, logp);
    ResourceArc::new(TrajectoryResource(Mutex::new(traj)))
}

/// Check if the trajectory has terminated (divergent or turning).
#[rustler::nif]
fn is_terminated(traj_ref: ResourceArc<TrajectoryResource>) -> bool {
    let traj = traj_ref.0.lock().unwrap();
    traj.is_terminated()
}

/// Get the endpoint as raw f64 binaries.
#[rustler::nif]
fn get_endpoint_bin<'a>(env: Env<'a>, traj_ref: ResourceArc<TrajectoryResource>, go_right: bool) -> (Binary<'a>, Binary<'a>, Binary<'a>) {
    let traj = traj_ref.0.lock().unwrap();
    let (q, p, grad) = traj.get_endpoint(go_right);
    (
        f64_slice_to_binary(env, q),
        f64_slice_to_binary(env, p),
        f64_slice_to_binary(env, grad),
    )
}

/// Build a subtree from pre-computed binary states AND merge into trajectory.
///
/// Hot path: one NIF call per doubling iteration. All data as raw binaries.
#[rustler::nif(schedule = "DirtyCpu")]
fn build_and_merge_bin(
    traj_ref: ResourceArc<TrajectoryResource>,
    all_q_bin: Binary,
    all_p_bin: Binary,
    all_logp_bin: Binary,
    all_grad_bin: Binary,
    inv_mass_bin: Binary,
    joint_logp_0: f64,
    depth: usize,
    d: usize,
    go_right: bool,
    rng_seed: u64,
) -> Atom {
    let states = PrecomputedStates {
        all_q: bytes_to_f64_vec(all_q_bin.as_slice()),
        all_p: bytes_to_f64_vec(all_p_bin.as_slice()),
        all_logp: bytes_to_f64_vec(all_logp_bin.as_slice()),
        all_grad: bytes_to_f64_vec(all_grad_bin.as_slice()),
        d,
    };
    let inv_mass = bytes_to_f64_vec(inv_mass_bin.as_slice());

    let mut rng = Xoshiro256StarStar::seed_from_u64(rng_seed);
    let mut counter = 0usize;

    let subtree = tree::build_subtree(
        &states, &inv_mass, joint_logp_0, depth, go_right, &mut counter, &mut rng,
    );

    let mut traj = traj_ref.0.lock().unwrap();
    tree::merge_into_trajectory(&mut traj, subtree, go_right, &inv_mass, &mut rng);

    rustler::types::atom::ok()
}

/// Build a subtree from pre-computed binary states and return the full subtree as a map.
///
/// Unlike `build_and_merge_bin` which merges into a trajectory, this returns the
/// complete subtree state so Elixir can handle the outer-loop merge. This is used
/// as a drop-in replacement for `build_subtree_cached` in tree.ex.
#[rustler::nif(schedule = "DirtyCpu")]
fn build_subtree_bin<'a>(
    env: Env<'a>,
    all_q_bin: Binary<'a>,
    all_p_bin: Binary<'a>,
    all_logp_bin: Binary<'a>,
    all_grad_bin: Binary<'a>,
    inv_mass_bin: Binary<'a>,
    joint_logp_0: f64,
    depth: usize,
    d: usize,
    going_right: bool,
    rng_seed: u64,
) -> NifResult<Term<'a>> {
    let states = PrecomputedStates {
        all_q: bytes_to_f64_vec(all_q_bin.as_slice()),
        all_p: bytes_to_f64_vec(all_p_bin.as_slice()),
        all_logp: bytes_to_f64_vec(all_logp_bin.as_slice()),
        all_grad: bytes_to_f64_vec(all_grad_bin.as_slice()),
        d,
    };
    let inv_mass = bytes_to_f64_vec(inv_mass_bin.as_slice());

    let mut rng = Xoshiro256StarStar::seed_from_u64(rng_seed);
    let mut counter = 0usize;

    let node = tree::build_subtree(
        &states, &inv_mass, joint_logp_0, depth, going_right, &mut counter, &mut rng,
    );

    // Encode full TreeNode as a map with binary position data
    let map = rustler::Term::map_new(env);
    let map = map.map_put(
        Atom::from_str(env, "q_left_bin")?.encode(env),
        f64_slice_to_binary(env, &node.q_left).encode(env),
    )?;
    let map = map.map_put(
        Atom::from_str(env, "p_left_bin")?.encode(env),
        f64_slice_to_binary(env, &node.p_left).encode(env),
    )?;
    let map = map.map_put(
        Atom::from_str(env, "grad_left_bin")?.encode(env),
        f64_slice_to_binary(env, &node.grad_left).encode(env),
    )?;
    let map = map.map_put(
        Atom::from_str(env, "q_right_bin")?.encode(env),
        f64_slice_to_binary(env, &node.q_right).encode(env),
    )?;
    let map = map.map_put(
        Atom::from_str(env, "p_right_bin")?.encode(env),
        f64_slice_to_binary(env, &node.p_right).encode(env),
    )?;
    let map = map.map_put(
        Atom::from_str(env, "grad_right_bin")?.encode(env),
        f64_slice_to_binary(env, &node.grad_right).encode(env),
    )?;
    let map = map.map_put(
        Atom::from_str(env, "q_prop_bin")?.encode(env),
        f64_slice_to_binary(env, &node.q_prop).encode(env),
    )?;
    let map = map.map_put(
        Atom::from_str(env, "logp_prop")?.encode(env),
        node.logp_prop.encode(env),
    )?;
    let map = map.map_put(
        Atom::from_str(env, "grad_prop_bin")?.encode(env),
        f64_slice_to_binary(env, &node.grad_prop).encode(env),
    )?;
    let map = map.map_put(
        Atom::from_str(env, "log_sum_weight")?.encode(env),
        node.log_sum_weight.encode(env),
    )?;
    let map = map.map_put(
        Atom::from_str(env, "n_steps")?.encode(env),
        node.n_steps.encode(env),
    )?;
    let map = map.map_put(
        Atom::from_str(env, "divergent")?.encode(env),
        node.divergent.encode(env),
    )?;
    let map = map.map_put(
        Atom::from_str(env, "accept_sum")?.encode(env),
        node.accept_sum.encode(env),
    )?;
    let map = map.map_put(
        Atom::from_str(env, "turning")?.encode(env),
        node.turning.encode(env),
    )?;
    let map = map.map_put(
        Atom::from_str(env, "depth")?.encode(env),
        node.depth.encode(env),
    )?;
    let map = map.map_put(
        Atom::from_str(env, "rho_bin")?.encode(env),
        f64_slice_to_binary(env, &node.rho).encode(env),
    )?;

    Ok(map)
}

/// Build the full NUTS tree in a single NIF call from pre-computed chains.
///
/// Takes initial state (q0, p0, grad0, logp0), forward and backward leapfrog
/// chains as binaries, and builds the entire doubling tree in Rust.
/// Returns a result map with q_bin, logp, grad_bin, n_steps, divergent, accept_sum, depth.
#[rustler::nif(schedule = "DirtyCpu")]
fn build_full_tree_bin<'a>(
    env: Env<'a>,
    q0_bin: Binary<'a>,
    p0_bin: Binary<'a>,
    grad0_bin: Binary<'a>,
    logp0: f64,
    fwd_q_bin: Binary<'a>,
    fwd_p_bin: Binary<'a>,
    fwd_logp_bin: Binary<'a>,
    fwd_grad_bin: Binary<'a>,
    bwd_q_bin: Binary<'a>,
    bwd_p_bin: Binary<'a>,
    bwd_logp_bin: Binary<'a>,
    bwd_grad_bin: Binary<'a>,
    inv_mass_bin: Binary<'a>,
    joint_logp_0: f64,
    max_depth: usize,
    d: usize,
    rng_seed: u64,
) -> NifResult<Term<'a>> {
    let q0 = bytes_to_f64_vec(q0_bin.as_slice());
    let p0 = bytes_to_f64_vec(p0_bin.as_slice());
    let grad0 = bytes_to_f64_vec(grad0_bin.as_slice());

    let fwd_states = PrecomputedStates {
        all_q: bytes_to_f64_vec(fwd_q_bin.as_slice()),
        all_p: bytes_to_f64_vec(fwd_p_bin.as_slice()),
        all_logp: bytes_to_f64_vec(fwd_logp_bin.as_slice()),
        all_grad: bytes_to_f64_vec(fwd_grad_bin.as_slice()),
        d,
    };

    let bwd_states = PrecomputedStates {
        all_q: bytes_to_f64_vec(bwd_q_bin.as_slice()),
        all_p: bytes_to_f64_vec(bwd_p_bin.as_slice()),
        all_logp: bytes_to_f64_vec(bwd_logp_bin.as_slice()),
        all_grad: bytes_to_f64_vec(bwd_grad_bin.as_slice()),
        d,
    };

    let inv_mass = bytes_to_f64_vec(inv_mass_bin.as_slice());

    let mut rng = Xoshiro256StarStar::seed_from_u64(rng_seed);

    let result = tree::build_full_tree(
        q0, p0, grad0, logp0,
        &fwd_states, &bwd_states,
        &inv_mass, joint_logp_0, max_depth, &mut rng,
    );

    // Encode result as map (same format as get_result_bin)
    let map = rustler::Term::map_new(env);
    let map = map.map_put(
        Atom::from_str(env, "q_bin")?.encode(env),
        f64_slice_to_binary(env, &result.q_prop).encode(env),
    )?;
    let map = map.map_put(
        Atom::from_str(env, "logp")?.encode(env),
        result.logp_prop.encode(env),
    )?;
    let map = map.map_put(
        Atom::from_str(env, "grad_bin")?.encode(env),
        f64_slice_to_binary(env, &result.grad_prop).encode(env),
    )?;
    let map = map.map_put(
        Atom::from_str(env, "n_steps")?.encode(env),
        result.n_steps.encode(env),
    )?;
    let map = map.map_put(
        Atom::from_str(env, "divergent")?.encode(env),
        result.divergent.encode(env),
    )?;
    let map = map.map_put(
        Atom::from_str(env, "accept_sum")?.encode(env),
        result.accept_sum.encode(env),
    )?;
    let map = map.map_put(
        Atom::from_str(env, "depth")?.encode(env),
        result.depth.encode(env),
    )?;

    Ok(map)
}

/// Extract the final result with binary q/grad.
#[rustler::nif]
fn get_result_bin(env: Env, traj_ref: ResourceArc<TrajectoryResource>) -> NifResult<Term> {
    let traj = traj_ref.0.lock().unwrap();
    let result = tree::trajectory_to_result(&traj);

    let map = rustler::Term::map_new(env);
    let map = map.map_put(
        rustler::types::atom::Atom::from_str(env, "q_bin")?.encode(env),
        f64_slice_to_binary(env, &result.q_prop).encode(env),
    )?;
    let map = map.map_put(
        rustler::types::atom::Atom::from_str(env, "logp")?.encode(env),
        result.logp_prop.encode(env),
    )?;
    let map = map.map_put(
        rustler::types::atom::Atom::from_str(env, "grad_bin")?.encode(env),
        f64_slice_to_binary(env, &result.grad_prop).encode(env),
    )?;
    let map = map.map_put(
        rustler::types::atom::Atom::from_str(env, "n_steps")?.encode(env),
        result.n_steps.encode(env),
    )?;
    let map = map.map_put(
        rustler::types::atom::Atom::from_str(env, "divergent")?.encode(env),
        result.divergent.encode(env),
    )?;
    let map = map.map_put(
        rustler::types::atom::Atom::from_str(env, "accept_sum")?.encode(env),
        result.accept_sum.encode(env),
    )?;
    let map = map.map_put(
        rustler::types::atom::Atom::from_str(env, "depth")?.encode(env),
        result.depth.encode(env),
    )?;

    Ok(map)
}

// --- Legacy list-based API (kept for tests) ---

#[rustler::nif(schedule = "DirtyCpu")]
fn init_trajectory(
    q: Vec<f64>,
    p: Vec<f64>,
    grad: Vec<f64>,
    logp: f64,
) -> ResourceArc<TrajectoryResource> {
    let traj = Trajectory::new(q, p, grad, logp);
    ResourceArc::new(TrajectoryResource(Mutex::new(traj)))
}

#[rustler::nif]
fn get_endpoint(traj_ref: ResourceArc<TrajectoryResource>, go_right: bool) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let traj = traj_ref.0.lock().unwrap();
    let (q, p, grad) = traj.get_endpoint(go_right);
    (q.to_vec(), p.to_vec(), grad.to_vec())
}

#[rustler::nif(schedule = "DirtyCpu")]
fn build_and_merge(
    traj_ref: ResourceArc<TrajectoryResource>,
    all_q: Vec<f64>,
    all_p: Vec<f64>,
    all_logp: Vec<f64>,
    all_grad: Vec<f64>,
    inv_mass: Vec<f64>,
    joint_logp_0: f64,
    depth: usize,
    d: usize,
    go_right: bool,
    rng_seed: u64,
) -> Atom {
    let states = PrecomputedStates {
        all_q,
        all_p,
        all_logp,
        all_grad,
        d,
    };

    let mut rng = Xoshiro256StarStar::seed_from_u64(rng_seed);
    let mut counter = 0usize;

    let subtree = tree::build_subtree(
        &states, &inv_mass, joint_logp_0, depth, go_right, &mut counter, &mut rng,
    );

    let mut traj = traj_ref.0.lock().unwrap();
    tree::merge_into_trajectory(&mut traj, subtree, go_right, &inv_mass, &mut rng);

    rustler::types::atom::ok()
}

#[rustler::nif]
fn get_result(env: Env, traj_ref: ResourceArc<TrajectoryResource>) -> NifResult<Term> {
    let traj = traj_ref.0.lock().unwrap();
    let result = tree::trajectory_to_result(&traj);

    let map = rustler::Term::map_new(env);
    let map = map.map_put(
        rustler::types::atom::Atom::from_str(env, "q")?.encode(env),
        result.q_prop.encode(env),
    )?;
    let map = map.map_put(
        rustler::types::atom::Atom::from_str(env, "logp")?.encode(env),
        result.logp_prop.encode(env),
    )?;
    let map = map.map_put(
        rustler::types::atom::Atom::from_str(env, "grad")?.encode(env),
        result.grad_prop.encode(env),
    )?;
    let map = map.map_put(
        rustler::types::atom::Atom::from_str(env, "n_steps")?.encode(env),
        result.n_steps.encode(env),
    )?;
    let map = map.map_put(
        rustler::types::atom::Atom::from_str(env, "divergent")?.encode(env),
        result.divergent.encode(env),
    )?;
    let map = map.map_put(
        rustler::types::atom::Atom::from_str(env, "accept_sum")?.encode(env),
        result.accept_sum.encode(env),
    )?;
    let map = map.map_put(
        rustler::types::atom::Atom::from_str(env, "depth")?.encode(env),
        result.depth.encode(env),
    )?;

    Ok(map)
}

#[allow(non_local_definitions)]
fn on_load(env: Env, _info: Term) -> bool {
    let _ = rustler::resource!(TrajectoryResource, env);
    true
}

rustler::init!("Elixir.Exmc.NUTS.NativeTree", load = on_load);
