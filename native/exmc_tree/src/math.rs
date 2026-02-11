/// Numerically stable log-sum-exp for two values.
/// Handles -inf edge cases (both -inf => -inf).
pub fn log_sum_exp(a: f64, b: f64) -> f64 {
    let m = a.max(b);
    if m == f64::NEG_INFINITY {
        f64::NEG_INFINITY
    } else {
        m + ((a - m).exp() + (b - m).exp()).ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_sum_exp_basic() {
        let result = log_sum_exp(0.0_f64.ln(), 0.0_f64.ln());
        // ln(e^ln(0) + e^ln(0)) — but ln(0) = -inf, so result = -inf
        // Let's test with actual values
        let result = log_sum_exp(1.0, 2.0);
        let expected = (1.0_f64.exp() + 2.0_f64.exp()).ln();
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_log_sum_exp_neg_inf() {
        assert_eq!(log_sum_exp(f64::NEG_INFINITY, f64::NEG_INFINITY), f64::NEG_INFINITY);
    }

    #[test]
    fn test_log_sum_exp_one_neg_inf() {
        let result = log_sum_exp(1.0, f64::NEG_INFINITY);
        assert!((result - 1.0).abs() < 1e-10);
    }
}
