def solve_with_lu(factorization_func, A, b):
    """
    Solves Ax = b using a specified LU factorization function.
    """
    print(f"--- Solving with {factorization_func.__name__} ---")
    try:
        L, U = factorization_func(A)
        print("L:\n", np.round(L, 4))
        print("U:\n", np.round(U, 4))
        
        # Step 1: Solve Ly = b
        y = forward_substitution(L, b)
        
        # Step 2: Solve Ux = y
        x = backward_substitution(U, y)
        return x
    except np.linalg.LinAlgError as e:
        print(e)
        return None

# Example Matrix (Doolittle and Crout)
A = np.array([[2., 1., 1.], [4., 3., 3.], [8., 7., 9.]])
b = np.array([6., 14., 36.])

x_doolittle = solve_with_lu(doolittle_lu, A, b)
x_crout = solve_with_lu(crout_lu, A, b)

print("\nSolution (Doolittle):", np.round(x_doolittle, 4))
print("Solution (Crout):", np.round(x_crout, 4))
print("Verification (numpy.linalg.solve):", np.round(np.linalg.solve(A, b), 4))


# Example SPD Matrix (Cholesky)
A_spd = np.array([[4., 2., -2.], [2., 5., 5.], [-2., 5., 14.]])
b_spd = np.array([6., 27., 45.])

print("\n--- Solving with Cholesky ---")
try:
    L_cholesky = cholesky(A_spd)
    print("L (Cholesky):\n", np.round(L_cholesky, 4))
    
    # For A=LL^T, we solve Ly=b and then L^T x = y
    y = forward_substitution(L_cholesky, b_spd)
    x_cholesky = backward_substitution(L_cholesky.T, y) # Note the transpose!
    
    print("\nSolution (Cholesky):", np.round(x_cholesky, 4))
    print("Verification (numpy.linalg.solve):", np.round(np.linalg.solve(A_spd, b_spd), 4))
except np.linalg.LinAlgError as e:
    print(e)