/// A node in the NUTS binary tree. Stores left/right endpoints, current proposal,
/// and tree statistics.
///
/// All position/momentum/gradient data are flat Vec<f64> of length `d`.
#[derive(Clone)]
pub struct TreeNode {
    // Left endpoint
    pub q_left: Vec<f64>,
    pub p_left: Vec<f64>,
    pub grad_left: Vec<f64>,

    // Right endpoint
    pub q_right: Vec<f64>,
    pub p_right: Vec<f64>,
    pub grad_right: Vec<f64>,

    // Current proposal (selected via multinomial)
    pub q_prop: Vec<f64>,
    pub logp_prop: f64,
    pub grad_prop: Vec<f64>,

    // Cumulative momentum sum (ρ = Σ p_i over all leaves in this subtree).
    // Used for the generalized U-turn criterion (Betancourt 2017).
    pub rho: Vec<f64>,

    // Tree statistics
    pub log_sum_weight: f64,
    pub n_steps: usize,
    pub divergent: bool,
    pub accept_sum: f64,
    pub turning: bool,
    pub depth: usize,
}

/// Pre-computed leapfrog states from multi_step_fn.
/// All arrays are row-major: element [i][j] = all_x[i * d + j].
pub struct PrecomputedStates {
    pub all_q: Vec<f64>,     // n_steps * d
    pub all_p: Vec<f64>,     // n_steps * d
    pub all_logp: Vec<f64>,  // n_steps
    pub all_grad: Vec<f64>,  // n_steps * d
    pub d: usize,
}

impl PrecomputedStates {
    /// Get the i-th state as slices.
    #[inline]
    pub fn get_q(&self, i: usize) -> &[f64] {
        let start = i * self.d;
        &self.all_q[start..start + self.d]
    }

    #[inline]
    pub fn get_p(&self, i: usize) -> &[f64] {
        let start = i * self.d;
        &self.all_p[start..start + self.d]
    }

    #[inline]
    pub fn get_logp(&self, i: usize) -> f64 {
        self.all_logp[i]
    }

    #[inline]
    pub fn get_grad(&self, i: usize) -> &[f64] {
        let start = i * self.d;
        &self.all_grad[start..start + self.d]
    }

    /// Number of pre-computed states.
    #[inline]
    pub fn len(&self) -> usize {
        self.all_logp.len()
    }

    /// Create a sub-range view from offset..offset+count.
    pub fn slice(&self, offset: usize, count: usize) -> PrecomputedStates {
        let d = self.d;
        PrecomputedStates {
            all_q: self.all_q[offset * d..(offset + count) * d].to_vec(),
            all_p: self.all_p[offset * d..(offset + count) * d].to_vec(),
            all_logp: self.all_logp[offset..offset + count].to_vec(),
            all_grad: self.all_grad[offset * d..(offset + count) * d].to_vec(),
            d,
        }
    }
}

/// Result returned from build_subtree to Elixir.
pub struct TreeResult {
    pub q_prop: Vec<f64>,
    pub logp_prop: f64,
    pub grad_prop: Vec<f64>,
    pub n_steps: usize,
    pub divergent: bool,
    pub accept_sum: f64,
    pub depth: usize,
}

/// The overall trajectory state, held as a ResourceArc in Elixir.
/// Tracks the full trajectory (left/right endpoints) and the running
/// proposal across all doubling iterations.
pub struct Trajectory {
    // Left endpoint
    pub q_left: Vec<f64>,
    pub p_left: Vec<f64>,
    pub grad_left: Vec<f64>,

    // Right endpoint
    pub q_right: Vec<f64>,
    pub p_right: Vec<f64>,
    pub grad_right: Vec<f64>,

    // Current proposal
    pub q_prop: Vec<f64>,
    pub logp_prop: f64,
    pub grad_prop: Vec<f64>,

    // Cumulative momentum sum (ρ = Σ p_i over all trajectory points)
    pub rho: Vec<f64>,

    // Running statistics
    pub log_sum_weight: f64,
    pub n_steps: usize,
    pub divergent: bool,
    pub accept_sum: f64,
    pub turning: bool,
    pub depth: usize,

    // Dimension (used for debug assertions)
    #[allow(dead_code)]
    pub d: usize,
}

impl Trajectory {
    pub fn new(q: Vec<f64>, p: Vec<f64>, grad: Vec<f64>, logp: f64) -> Self {
        let d = q.len();
        let rho = p.clone();
        Trajectory {
            q_left: q.clone(),
            p_left: p.clone(),
            grad_left: grad.clone(),
            q_right: q.clone(),
            p_right: p.clone(),
            grad_right: grad.clone(),
            q_prop: q,
            logp_prop: logp,
            grad_prop: grad,
            rho,
            log_sum_weight: 0.0,
            n_steps: 0,
            divergent: false,
            accept_sum: 0.0,
            turning: false,
            depth: 0,
            d,
        }
    }

    pub fn is_terminated(&self) -> bool {
        self.divergent || self.turning
    }

    /// Get the endpoint to continue building from (for doubling).
    pub fn get_endpoint(&self, go_right: bool) -> (&[f64], &[f64], &[f64]) {
        if go_right {
            (&self.q_right, &self.p_right, &self.grad_right)
        } else {
            (&self.q_left, &self.p_left, &self.grad_left)
        }
    }
}
