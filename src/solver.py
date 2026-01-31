import numpy as np
import math

def calculate_probability(U, r_vec, s_vec, S_matrix):
    """
    Calculates the probability of a specific output distribution for partially distinguishable bosons.
    
    Args:
        U (np.ndarray): Unitary scattering matrix (m x m).
        r_vec (list/array): Input population vector (length m).
        s_vec (list/array): Output population vector (length m).
        S_matrix (np.ndarray): Distinguishability matrix (n x n).
        
    Returns:
        float: The probability of the transition.
    """
    # 1. Expand input/output vectors to index lists
    # r_vec indicates how many particles are in each input port.
    L_in = []
    for port_idx, count in enumerate(r_vec):
        L_in.extend([port_idx] * count)
    
    # s_vec indicates how many particles are in each output port.
    L_out = []
    for port_idx, count in enumerate(s_vec):
        L_out.extend([port_idx] * count)
        
    n = len(L_in)
    if len(L_out) != n:
        raise ValueError(f"Number of input particles ({n}) does not match output particles ({len(L_out)})")
    
    # 2. Build Effective Matrix M (n x n)
    # M[i, j] takes the element from U corresponding to the input port of particle i
    # and the output port of particle j.
    M = np.zeros((n, n), dtype=complex)
    for i in range(n):
        for j in range(n):
            M[i, j] = U[L_in[i], L_out[j]]
            
    # 3. Build Tensor W (n x n x n)
    # W[k, l, j] = M[k, j] * conj(M[l, j]) * S[l, k]
    # Note on indices from TASK.md vs PDF:
    # TASK.md: W[k, l, j] where k is row (direct), l is row (conjugate), j is column (output)
    # S[l, k] is the distinguishability between particle k and l.
    W = np.zeros((n, n, n), dtype=complex)
    for k in range(n):
        for l in range(n):
            # S is n x n, corresponding to the n particles.
            factor = S_matrix[l, k] 
            for j in range(n):
                W[k, l, j] = M[k, j] * np.conj(M[l, j]) * factor
                
    # 4. Ryser's Algorithm for Tensor Permanent
    total_sum = 0.0 + 0.0j
    
    # Iterate through all subsets A and B (represented by bitmasks)
    # Range is 0 to 2^n - 1. 
    # TASK.md suggests iterating 1 to 2^n because empty set contributions are 0,
    # but standard Ryser usually includes 0 with (-1)^0 term.
    # However, for permanent, empty set usually implies count of 0, but here we iterate subsets of indices.
    # Let's follow TASK.md loop range (0 to 2^n is safer to cover all logic, mask=0 gives empty set).
    # TASK.md says: "Range is 1 to 2^n because empty set contributions are 0"
    # Let's implement full range 0 to 2^n for correctness of the formula structure, 
    # checking if empty set yields 0 product.
    
    limit = 1 << n
    
    for mask_A in range(limit):
        # Extract indices for set A
        idx_A = [i for i in range(n) if (mask_A & (1 << i))]
        # Sign calculation: (-1)^|A|
        sign_A = -1 if (len(idx_A) % 2 == 1) else 1
        
        for mask_B in range(limit):
            idx_B = [i for i in range(n) if (mask_B & (1 << i))]
            # Sign calculation: (-1)^|B|
            sign_B = -1 if (len(idx_B) % 2 == 1) else 1
            
            # Combined sign
            Sign = sign_A * sign_B
            
            # Calculate Product term
            # Product over j=1..n of Sum(W[k,l,j]) where k in A, l in B
            prod_term = 1.0 + 0.0j
            
            for j in range(n):
                inner_sum = 0.0 + 0.0j
                
                # Optimized inner loops?
                # For small n (<=7), strict loops are fine. 
                # Vectorization is possible but let's stick to clarity first.
                for k in idx_A:
                    for l in idx_B:
                        inner_sum += W[k, l, j]
                
                prod_term *= inner_sum
                
                # Optimization: if any inner_sum is 0, prod is 0
                if inner_sum == 0:
                    break
            
            total_sum += Sign * prod_term
            
    # 5. Normalization
    normalization = 1.0
    for k in r_vec: 
        normalization *= math.factorial(k)
    for k in s_vec: 
        normalization *= math.factorial(k)
    
    # Correction factor: (-1)^n
    # The formula in TASK.md is: final_prob = total_sum.real * ((-1)**n) / normalization
    
    final_prob = total_sum.real * ((-1)**n) / normalization
    
    # Probability cannot be negative (computational errors might give -1e-16)
    return max(0.0, final_prob)
