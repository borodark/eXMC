use rand::Rng;
use rand_xoshiro::Xoshiro256StarStar;

use crate::math::log_sum_exp;
use crate::types::{PrecomputedStates, Trajectory, TreeNode, TreeResult};
use crate::uturn::check_uturn;

/// Build a subtree of given depth from pre-computed leapfrog states.
///
/// `counter` tracks which pre-computed state to consume next (incremented at each leaf).
/// `going_right` indicates the direction (epsilon > 0 means right).
/// `rng` provides randomness for multinomial proposal selection.
///
/// This is a 1:1 translation of tree.ex's build_subtree, but operates on flat Vec<f64>
/// instead of Nx tensors + Elixir maps.
pub fn build_subtree(
    states: &PrecomputedStates,
    inv_mass: &[f64],
    joint_logp_0: f64,
    depth: usize,
    going_right: bool,
    counter: &mut usize,
    rng: &mut Xoshiro256StarStar,
) -> TreeNode {
    if depth == 0 {
        // Base case: consume one pre-computed state
        build_leaf(states, inv_mass, joint_logp_0, counter)
    } else {
        // Recursive: build two half-subtrees and merge
        let half_depth = depth - 1;

        let first = build_subtree(states, inv_mass, joint_logp_0, half_depth, going_right, counter, rng);

        if first.divergent || first.turning {
            return first;
        }

        let second = build_subtree(states, inv_mass, joint_logp_0, half_depth, going_right, counter, rng);

        merge_subtrees(first, second, going_right, inv_mass, rng)
    }
}

/// Build a leaf node from the next pre-computed state.
fn build_leaf(
    states: &PrecomputedStates,
    inv_mass: &[f64],
    joint_logp_0: f64,
    counter: &mut usize,
) -> TreeNode {
    let idx = *counter;
    *counter += 1;

    let q = states.get_q(idx).to_vec();
    let p = states.get_p(idx).to_vec();
    let logp = states.get_logp(idx);
    let grad = states.get_grad(idx).to_vec();

    // Compute joint_logp = logp - 0.5 * p^T * inv_mass * p
    let ke: f64 = p.iter()
        .zip(inv_mass.iter())
        .map(|(pi, mi)| 0.5 * pi * mi * pi)
        .sum();
    let joint_logp = logp - ke;

    // Divergence and acceptance
    let (divergent, log_weight, accept_prob) = if joint_logp.is_finite() {
        let d = joint_logp - joint_logp_0;
        let divergent = d < -1000.0;
        let log_weight = d;
        let accept_prob = d.min(0.0).exp().min(1.0);
        (divergent, log_weight, accept_prob)
    } else {
        (true, -1001.0, 0.0)
    };

    TreeNode {
        q_left: q.clone(),
        p_left: p.clone(),
        grad_left: grad.clone(),
        q_right: q.clone(),
        p_right: p.clone(),
        grad_right: grad.clone(),
        q_prop: q,
        logp_prop: logp,
        grad_prop: grad,
        rho: p,
        log_sum_weight: log_weight,
        n_steps: 1,
        divergent,
        accept_sum: accept_prob,
        turning: false,
        depth: 0,
    }
}

/// Merge two subtrees (within a doubling level).
///
/// Follows the Betancourt (2017) multinomial NUTS variant:
/// - Proposal selected via multinomial weighting (log-sum-exp)
/// - Endpoints determined by direction (going_right)
/// - U-turn check on merged endpoints
fn merge_subtrees(
    first: TreeNode,
    second: TreeNode,
    going_right: bool,
    inv_mass: &[f64],
    rng: &mut Xoshiro256StarStar,
) -> TreeNode {
    let combined_log_weight = log_sum_exp(first.log_sum_weight, second.log_sum_weight);
    let combined_n_steps = first.n_steps + second.n_steps;
    let combined_accept_sum = first.accept_sum + second.accept_sum;
    let combined_divergent = first.divergent || second.divergent;

    // Multinomial: accept second's proposal with prob exp(second.log_weight - combined)
    let rand_val: f64 = rng.gen();
    let accept_prob = (second.log_sum_weight - combined_log_weight).exp();
    let use_second = rand_val < accept_prob;

    let (q_prop, logp_prop, grad_prop) = if use_second {
        (second.q_prop.clone(), second.logp_prop, second.grad_prop.clone())
    } else {
        (first.q_prop.clone(), first.logp_prop, first.grad_prop.clone())
    };

    // Cumulative momentum sum: ρ = ρ_first + ρ_second
    let rho: Vec<f64> = first.rho.iter().zip(second.rho.iter()).map(|(a, b)| a + b).collect();

    // Sub-trajectory U-turn checks (PyMC-style, before endpoint destructuring)
    // Only when children have > 1 leaf (depth > 0)
    let sub_turning = if !combined_divergent && !second.turning && first.depth > 0 {
        let (left_rho, right_rho, left_p_left, left_p_right, right_p_left, right_p_right) =
            if going_right {
                (&first.rho, &second.rho, &first.p_left, &first.p_right, &second.p_left, &second.p_right)
            } else {
                (&second.rho, &first.rho, &second.p_left, &second.p_right, &first.p_left, &first.p_right)
            };

        // Check 2: left sub-trajectory + first point of right sub-trajectory
        let partial_rho_2: Vec<f64> = left_rho.iter().zip(right_p_left.iter()).map(|(a, b)| a + b).collect();
        if check_uturn(&partial_rho_2, left_p_left, right_p_left, inv_mass) {
            true
        } else {
            // Check 3: last point of left sub-trajectory + right sub-trajectory
            let partial_rho_3: Vec<f64> = left_p_right.iter().zip(right_rho.iter()).map(|(a, b)| a + b).collect();
            check_uturn(&partial_rho_3, left_p_right, right_p_right, inv_mass)
        }
    } else {
        false
    };

    // Endpoints depend on direction
    let (q_left, p_left, grad_left, q_right, p_right, grad_right) = if going_right {
        (
            first.q_left, first.p_left, first.grad_left,
            second.q_right, second.p_right, second.grad_right,
        )
    } else {
        (
            second.q_left, second.p_left, second.grad_left,
            first.q_right, first.p_right, first.grad_right,
        )
    };

    // U-turn checks (Betancourt 2017): ρ · (M^{-1} p±) < 0
    let turning = combined_divergent
        || second.turning
        || sub_turning
        || check_uturn(&rho, &p_left, &p_right, inv_mass);

    TreeNode {
        q_left,
        p_left,
        grad_left,
        q_right,
        p_right,
        grad_right,
        q_prop,
        logp_prop,
        grad_prop,
        rho,
        log_sum_weight: combined_log_weight,
        n_steps: combined_n_steps,
        divergent: combined_divergent,
        accept_sum: combined_accept_sum,
        turning,
        depth: first.depth.max(second.depth) + 1,
    }
}

/// Merge a subtree into the overall trajectory (top-level doubling merge).
///
/// This corresponds to merge_trajectories in tree.ex.
pub fn merge_into_trajectory(
    traj: &mut Trajectory,
    subtree: TreeNode,
    go_right: bool,
    inv_mass: &[f64],
    rng: &mut Xoshiro256StarStar,
) {
    let combined_log_weight = log_sum_exp(traj.log_sum_weight, subtree.log_sum_weight);
    let combined_n_steps = traj.n_steps + subtree.n_steps;
    let combined_accept_sum = traj.accept_sum + subtree.accept_sum;
    let combined_divergent = traj.divergent || subtree.divergent;
    let subtree_turning = subtree.turning;

    // Sub-trajectory U-turn checks (before any mutations)
    let sub_turning = if !combined_divergent && !subtree_turning {
        let (left_rho, right_rho, left_p_left, left_p_right, right_p_left, right_p_right) =
            if go_right {
                (&traj.rho[..], &subtree.rho[..], &traj.p_left[..], &traj.p_right[..], &subtree.p_left[..], &subtree.p_right[..])
            } else {
                (&subtree.rho[..], &traj.rho[..], &subtree.p_left[..], &subtree.p_right[..], &traj.p_left[..], &traj.p_right[..])
            };

        // Check 2: left sub-trajectory + first point of right sub-trajectory
        let partial_rho_2: Vec<f64> = left_rho.iter().zip(right_p_left.iter()).map(|(a, b)| a + b).collect();
        if check_uturn(&partial_rho_2, left_p_left, right_p_left, inv_mass) {
            true
        } else {
            // Check 3: last point of left sub-trajectory + right sub-trajectory
            let partial_rho_3: Vec<f64> = left_p_right.iter().zip(right_rho.iter()).map(|(a, b)| a + b).collect();
            check_uturn(&partial_rho_3, left_p_right, right_p_right, inv_mass)
        }
    } else {
        false
    };

    // Biased progressive sampling (Stan/PyMC): use OLD trajectory weight
    let rand_val: f64 = rng.gen();
    if rand_val.ln() < (subtree.log_sum_weight - traj.log_sum_weight) {
        traj.q_prop = subtree.q_prop;
        traj.logp_prop = subtree.logp_prop;
        traj.grad_prop = subtree.grad_prop;
    }

    // Accumulate momentum sum: ρ = ρ_traj + ρ_subtree
    for i in 0..traj.rho.len() {
        traj.rho[i] += subtree.rho[i];
    }

    // Update endpoints
    if go_right {
        traj.q_right = subtree.q_right;
        traj.p_right = subtree.p_right;
        traj.grad_right = subtree.grad_right;
    } else {
        traj.q_left = subtree.q_left;
        traj.p_left = subtree.p_left;
        traj.grad_left = subtree.grad_left;
    }

    // Full trajectory U-turn check (after mutations)
    let turning = combined_divergent
        || subtree_turning
        || sub_turning
        || check_uturn(&traj.rho, &traj.p_left, &traj.p_right, inv_mass);

    traj.log_sum_weight = combined_log_weight;
    traj.n_steps = combined_n_steps;
    traj.accept_sum = combined_accept_sum;
    traj.divergent = combined_divergent;
    traj.turning = turning;
    traj.depth += 1;
}

/// Build the full NUTS tree in a single call.
///
/// Takes pre-computed forward and backward leapfrog chains from q0 and builds the
/// entire doubling tree (direction choices, subtree construction, merges, U-turn
/// checks, termination) without returning to Elixir.
///
/// `fwd_states` contains states from q0 with +epsilon (indices 0..budget).
/// `bwd_states` contains states from q0 with -epsilon (indices 0..budget).
/// Direction choices consume from the appropriate chain.
pub fn build_full_tree(
    q0: Vec<f64>,
    p0: Vec<f64>,
    grad0: Vec<f64>,
    logp0: f64,
    fwd_states: &PrecomputedStates,
    bwd_states: &PrecomputedStates,
    inv_mass: &[f64],
    joint_logp_0: f64,
    max_depth: usize,
    rng: &mut Xoshiro256StarStar,
) -> TreeResult {
    let mut traj = Trajectory::new(q0, p0, grad0, logp0);
    let mut fwd_cursor: usize = 0;
    let mut bwd_cursor: usize = 0;

    for _depth in 0..max_depth {
        if traj.is_terminated() {
            break;
        }

        let go_right: bool = rng.gen::<f64>() > 0.5;
        let n_steps = 1usize << traj.depth;

        // Check if enough pre-computed states remain for this doubling level
        if go_right && fwd_cursor + n_steps > fwd_states.len() {
            break;
        }
        if !go_right && bwd_cursor + n_steps > bwd_states.len() {
            break;
        }

        let subtree = if go_right {
            let sub = fwd_states.slice(fwd_cursor, n_steps);
            let mut counter = 0;
            let s = build_subtree(&sub, inv_mass, joint_logp_0, traj.depth, go_right, &mut counter, rng);
            fwd_cursor += n_steps;
            s
        } else {
            let sub = bwd_states.slice(bwd_cursor, n_steps);
            let mut counter = 0;
            let s = build_subtree(&sub, inv_mass, joint_logp_0, traj.depth, go_right, &mut counter, rng);
            bwd_cursor += n_steps;
            s
        };

        merge_into_trajectory(&mut traj, subtree, go_right, inv_mass, rng);
    }

    trajectory_to_result(&traj)
}

/// Convert a trajectory into a final result for returning to Elixir.
pub fn trajectory_to_result(traj: &Trajectory) -> TreeResult {
    TreeResult {
        q_prop: traj.q_prop.clone(),
        logp_prop: traj.logp_prop,
        grad_prop: traj.grad_prop.clone(),
        n_steps: traj.n_steps,
        divergent: traj.divergent,
        accept_sum: traj.accept_sum,
        depth: traj.depth,
    }
}
