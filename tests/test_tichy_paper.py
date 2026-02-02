import pytest
import numpy as np
from src.solver import calculate_probability

def create_fourier_matrix(m):
    """
    Creates a unitary Fourier matrix of size m x m.
    U_jk = (1/sqrt(m)) * exp(i * 2*pi * j * k / m)
    using 0-based indexing for j, k.
    """
    U = np.zeros((m, m), dtype=complex)
    for j in range(m):
        for k in range(m):
            # Using the standard definition from quantum optics papers usually involved in Boson Sampling
            # Note: The paper Eq 33 uses 1/sqrt(n) but describes m modes. Normalization must be 1/sqrt(m).
            # The exponent factor depends on convention, but usually symmetric.
            U[j, k] = np.exp(2j * np.pi * j * k / m)
    return U / np.sqrt(m)

def test_perfect_suppression_section_III_E():
    """
    Reproduces the counter-intuitive example from Section III.E of arXiv:1410.7687v3.
    
    Setup:
    - m = 9 modes.
    - U = Fourier Matrix.
    - Input r: Particles at modes 1, 4, 7, 9 (1-based indices).
      Input vector r_vec has 1s at indices 0, 3, 6, 8.
    - Output s: Particles at modes 2, 3, 5, 9 (1-based indices).
      Output vector s_vec has 1s at indices 1, 2, 4, 8.
      
    Particles:
    - The first 3 particles (at 0, 3, 6) are identical to each other.
    - The 4th particle (at 8) has distinguishability 'x' with the others.
    
    Prediction:
    - The probability P(s) should be 0 for ANY value of x.
    """
    m = 9
    U = create_fourier_matrix(m)
    
    # Input: modes 1, 4, 7, 9 -> indices 0, 3, 6, 8
    r_vec = [0] * m
    for idx in [0, 3, 6, 8]:
        r_vec[idx] = 1
        
    # Output: modes 2, 3, 5, 9 -> indices 1, 2, 4, 8
    s_vec = [0] * m
    for idx in [1, 2, 4, 8]:
        s_vec[idx] = 1
        
    # Test for various values of x (degree of distinguishability)
    # x = 1 (Identical), x = 0 (Distinguishable), x = 0.5 (Partial)
    test_x_values = [0.0, 0.1, 0.5, 0.9, 1.0]
    
    for x in test_x_values:
        # Construct S matrix
        # Particles order in L_in will be based on r_vec:
        # Particle 0 -> mode 0
        # Particle 1 -> mode 3
        # Particle 2 -> mode 6
        # Particle 3 -> mode 8 (The distinguishable one)
        
        S = np.array([
            [1, 1, 1, x],
            [1, 1, 1, x],
            [1, 1, 1, x],
            [x, x, x, 1]
        ], dtype=complex)
        
        prob = calculate_probability(U, r_vec, s_vec, S)
        
        # Check if probability is 0 (within numerical precision)
        # Using a slightly larger tolerance because n=4, m=9 involves many float ops
        assert np.isclose(prob, 0.0, atol=1e-9), f"Failed for x={x}, P={prob}"

def test_full_bunching_law_section_IV_C():
    """
    Section IV.C: Bunching events.
    When all n particles bunch into a single output port j.
    P_bunch = perm(S) * P_dist
    """
    # Simple setup: 3 particles, random unitary
    n = 3
    m = 3
    # Fixed random seed for reproducibility
    np.random.seed(42)
    # Generate random unitary (QR decomposition of random matrix)
    Z = np.random.randn(m, m) + 1j * np.random.randn(m, m)
    Q, R = np.linalg.qr(Z)
    U = Q
    
    r_vec = [1, 1, 1]
    
    # Bunching event: all in output 0
    s_vec = [3, 0, 0]
    
    # Random S matrix
    # Must be Hermitian and positive semi-definite.
    # Let's construct one from vectors.
    vecs = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    # Normalize vectors
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    S = vecs @ vecs.T.conj()
    # Ensure diagonal is real 1.0 (it should be close by construction)
    for i in range(n):
        S[i, i] = 1.0
        
    # Calculate P_dist (S=I)
    S_dist = np.eye(n)
    p_dist = calculate_probability(U, r_vec, s_vec, S_dist)
    
    # Calculate P_S
    p_s = calculate_probability(U, r_vec, s_vec, S)
    
    # Calculate perm(S)
    # Since n=3 is small, we can use a naive permanent calculation or reusing the solver with U=I?
    # Actually, let's just compute permanent manually for 3x3
    # Perm(S) = sum over sigma of prod S_i,sigma(i)
    # Or reuse the logic: P_bunch / P_dist = perm(S) / 1!1!1! (since input is 1,1,1)
    # Wait, Eq 53: P_S(bunch) = perm(S) * prod(r_k!) * P_dist(bunch)
    # Here input is 1,1,1 so prod(r_k!) = 1.
    # So expected: P_S = perm(S) * P_dist.
    
    # Compute perm(S) manually for n=3
    # S = [[a, b, c], [d, e, f], [g, h, i]]
    # perm = aei + ahf + bdi + bgf + cdh + cge
    # Since S is Hermitian: d=b*, g=c*, h=f*
    perm_S = 0
    import itertools
    for sigma in itertools.permutations(range(n)):
        term = 1
        for i in range(n):
            term *= S[i, sigma[i]]
        perm_S += term
    
    # Check relation
    assert np.isclose(p_s, perm_S.real * p_dist, atol=1e-9)

if __name__ == "__main__":
    try:
        test_perfect_suppression_section_III_E()
        print("Test III.E Passed")
        test_full_bunching_law_section_IV_C()
        print("Test IV.C Passed")
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
