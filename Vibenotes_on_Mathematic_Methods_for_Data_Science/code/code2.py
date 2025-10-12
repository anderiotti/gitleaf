def gauss_seidel_solve(A, b, x0, max_iter, tol=1e-6):
    """
    Solves Ax = b using the Gauss-Seidel method. M = D - L, N = U.
    Requires an explicit sequential loop (forward substitution).
    """
    n = A.shape[0]
    x_old = np.array(x0, dtype=A.dtype)
    
    for k in range(1, max_iter + 1):
        x_new = np.copy(x_old)
        
        # Sequential update: The causal heart of the method
        for i in range(n):
            # 1. Calculate sum for j < i (L*x_new: using updated values)
            sum_L = np.dot(A[i, :i], x_new[:i])
            # 2. Calculate sum for j > i (U*x_old: using old values)
            sum_U = np.dot(A[i, i+1:], x_old[i+1:])
            
            # Update: x_i^(k+1) = (b_i - sum_L - sum_U) / A_ii
            x_new[i] = (b[i] - sum_L - sum_U) / A[i, i]
        
        # Relative Error Criterion (L_inf norm)
        error_norm = np.linalg.norm(x_new - x_old, np.inf)
        if error_norm < tol * np.linalg.norm(x_new, np.inf):
            # print(f"Gauss-Seidel converged in {k} iterations.")
            return x_new, k
            
        x_old = x_new
        
    # print(f"Gauss-Seidel failed to converge in {max_iter} iterations.")
    return x_old, max_iter