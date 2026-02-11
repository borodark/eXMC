/// Generalized U-turn check (Betancourt 2017) with diagonal mass matrix.
///
/// Uses the cumulative momentum sum ρ instead of endpoint displacement (q+ - q-).
/// Returns true if either endpoint velocity is anti-aligned with ρ, indicating
/// the trajectory has turned around.
///
/// Checks: ρ · (M^{-1} * p_left) < 0 OR ρ · (M^{-1} * p_right) < 0
pub fn check_uturn(
    rho: &[f64],
    p_left: &[f64],
    p_right: &[f64],
    inv_mass: &[f64],
) -> bool {
    let mut dot_right = 0.0;
    let mut dot_left = 0.0;

    for i in 0..rho.len() {
        let v = rho[i] * inv_mass[i];
        dot_right += v * p_right[i];
        dot_left += v * p_left[i];
    }

    dot_right < 0.0 || dot_left < 0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_uturn() {
        // Rho aligned with both endpoint momenta
        let rho = vec![2.0, 2.0];
        let p_left = vec![1.0, 1.0];
        let p_right = vec![1.0, 1.0];
        let inv_mass = vec![1.0, 1.0];
        assert!(!check_uturn(&rho, &p_left, &p_right, &inv_mass));
    }

    #[test]
    fn test_uturn() {
        // Right momentum pointing opposite to rho
        let rho = vec![2.0, 2.0];
        let p_left = vec![1.0, 1.0];
        let p_right = vec![-1.0, -1.0];
        let inv_mass = vec![1.0, 1.0];
        assert!(check_uturn(&rho, &p_left, &p_right, &inv_mass));
    }

    #[test]
    fn test_rho_vs_endpoint_difference() {
        // Case where rho and (q+ - q-) disagree due to non-uniform mass matrix.
        // With large inv_mass range, endpoint criterion is dominated by high-variance
        // components while rho criterion weights all components more uniformly.
        let rho = vec![3.0, 3.0]; // Sum of momenta
        let p_left = vec![1.0, 1.0];
        let p_right = vec![-0.5, 1.5]; // Slightly turned in first component
        let inv_mass = vec![10.0, 0.1]; // 100x range

        // rho · (M^{-1} p_right) = 3*10*(-0.5) + 3*0.1*1.5 = -15 + 0.45 = -14.55 → U-turn
        // This correctly detects the turn because rho is the average trajectory direction
        assert!(check_uturn(&rho, &p_left, &p_right, &inv_mass));
    }
}
