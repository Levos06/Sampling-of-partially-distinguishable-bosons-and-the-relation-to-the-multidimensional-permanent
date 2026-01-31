import pytest
import numpy as np
from src.solver import calculate_probability

def test_hong_ou_mandel_identical():
    """
    Test 1: Identical Bosons
    n=2, U=Hadamard, Input=[1,1], Output=[1,1].
    Expected Result: 0 (Destructive interference).
    """
    # Hadamard Matrix
    U = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    r_vec = [1, 1]
    s_vec = [1, 1]
    
    # S matrix for identical particles (all 1s)
    S = np.ones((2, 2))
    
    prob = calculate_probability(U, r_vec, s_vec, S)
    
    # Assert close to 0
    assert np.isclose(prob, 0.0, atol=1e-10)

def test_distinguishable_particles():
    """
    Test 2: Distinguishable Particles
    n=2, U=Hadamard, Input=[1,1], Output=[1,1].
    Expected Result: 0.5 (Classical probability).
    """
    U = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    r_vec = [1, 1]
    s_vec = [1, 1]
    
    # S matrix for distinguishable particles (Identity)
    S = np.eye(2)
    
    prob = calculate_probability(U, r_vec, s_vec, S)
    
    # Classical prob: |1/sqrt(2) * 1/sqrt(2)|^2 + ... 
    # Actually for distinguishable:
    # Particle 1 goes L, Particle 2 goes R: prob 1/2 * 1/2 = 1/4
    # Particle 1 goes R, Particle 2 goes L: prob 1/2 * 1/2 = 1/4
    # Total = 0.5
    assert np.isclose(prob, 0.5, atol=1e-10)

def test_partial_distinguishability():
    """
    Test 3: Partial Distinguishability
    n=2, U=Hadamard, Input=[1,1], Output=[1,1].
    S off-diagonal = x.
    Expected Result: 0.5 * (1 - x^2).
    """
    U = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    r_vec = [1, 1]
    s_vec = [1, 1]
    
    x = 0.5 # Example value
    S = np.array([[1.0, x], [x, 1.0]])
    
    prob = calculate_probability(U, r_vec, s_vec, S)
    
    expected = 0.5 * (1 - x**2)
    assert np.isclose(prob, expected, atol=1e-10)

def test_bunching_identical():
    """
    Extra Test: Bunching Identical
    Output [2,0] or [0,2].
    Prob should be 0.5 each.
    """
    U = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    r_vec = [1, 1]
    
    # Case [2, 0]
    s_vec_1 = [2, 0]
    S = np.ones((2, 2))
    prob_1 = calculate_probability(U, r_vec, s_vec_1, S)
    assert np.isclose(prob_1, 0.5, atol=1e-10)
    
    # Case [0, 2]
    s_vec_2 = [0, 2]
    prob_2 = calculate_probability(U, r_vec, s_vec_2, S)
    assert np.isclose(prob_2, 0.5, atol=1e-10)

if __name__ == "__main__":
    # Manual run if pytest not installed or for quick check
    try:
        test_hong_ou_mandel_identical()
        print("Test 1 Passed")
        test_distinguishable_particles()
        print("Test 2 Passed")
        test_partial_distinguishability()
        print("Test 3 Passed")
        test_bunching_identical()
        print("Test 4 Passed")
    except AssertionError as e:
        print(f"Test Failed: {e}")
    except Exception as e:
        print(f"Error: {e}")
