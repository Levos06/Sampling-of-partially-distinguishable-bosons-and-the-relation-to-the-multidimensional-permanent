import numpy as np
from src.solver import calculate_probability

def run_demo():
    print("=== Boson Sampling Simulation Demo ===")
    
    # Setup: 2-mode interferometer (Beam Splitter)
    U = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    print("\nUnitary Matrix (Beam Splitter):")
    print(U)
    
    r_vec = [1, 1] # One photon in each input
    s_vec = [1, 1] # Coincidence detection
    
    print(f"\nInput: {r_vec}")
    print(f"Output: {s_vec}")
    
    # 1. Identical
    S_id = np.ones((2, 2))
    p_id = calculate_probability(U, r_vec, s_vec, S_id)
    print(f"\n1. Identical Particles (S=1): P = {p_id:.4f} (Expected: 0.0000)")
    
    # 2. Distinguishable
    S_dist = np.eye(2)
    p_dist = calculate_probability(U, r_vec, s_vec, S_dist)
    print(f"2. Distinguishable Particles (S=I): P = {p_dist:.4f} (Expected: 0.5000)")
    
    # 3. Partial
    x = 0.5
    S_part = np.array([[1.0, x], [x, 1.0]])
    p_part = calculate_probability(U, r_vec, s_vec, S_part)
    expected = 0.5 * (1 - x**2)
    print(f"3. Partial Distinguishability (x={x}): P = {p_part:.4f} (Expected: {expected:.4f})")

if __name__ == "__main__":
    run_demo()
