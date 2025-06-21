import numpy as np
from typing import List, Tuple, Optional, Union

class RevisedSimplexSolver:
    def __init__(self, c: List[float], A: List[List[float]], b: List[float], eps: float = 1e-10):
        """
        Initialize the Revised Simplex solver.
        
        Parameters:
            c: Objective function coefficients (to maximize)
            A: Constraint matrix (each row: ≤ constraint)  
            b: Right-hand side of constraints
            eps: Numerical tolerance
        """
        self.c = np.array(c, dtype=float)
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float)
        self.eps = eps
        
        # Problem dimensions
        self.m = len(b)  # number of constraints
        self.n = len(c)  # number of original variables
        
        # Validation
        self._validate_inputs()
        
        # Extended problem with slack variables
        self.c_extended = np.hstack([self.c, np.zeros(self.m)])
        self.A_extended = np.hstack([self.A, np.eye(self.m)])
        self.n_extended = self.n + self.m
        
        # Initialize
        self._initialize()
    
    def _validate_inputs(self):
        """Validate input dimensions and feasibility conditions."""
        if self.A.shape[0] != self.m:
            raise ValueError("Inconsistent number of constraints")
        if self.A.shape[1] != self.n:
            raise ValueError("Inconsistent number of variables")
        if np.any(self.b < 0):
            raise ValueError("RHS must be non-negative for standard form")
    
    def _initialize(self):
        """Step 0: Initialize with slack variables as starting feasible basis."""
        # Starting basis: slack variables (columns n to n+m-1)
        self.basic_indices = list(range(self.n, self.n_extended))
        self.nonbasic_indices = list(range(self.n))
        
        # Basic column matrix B and its inverse B^(-1)
        self.B = self.A_extended[:, self.basic_indices].copy()
        self.B_inv = np.eye(self.m)  # Initially identity since B = I
        
        # Current solution
        self.x = np.zeros(self.n_extended)
        self.x[self.basic_indices] = self.b.copy()  # Basic variables = RHS
        
        # Solution index
        self.t = 0
        
        print("Step 0: Initialization complete")
        print(f"Initial basic variables: {self.basic_indices}")
        print(f"Initial solution: x_B = {self.x[self.basic_indices]}")
    
    def _pricing(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Step 1: Compute pricing vector and reduced costs.
        
        Returns:
            v: Pricing vector (solution to v^T B = c_B^T)
            reduced_costs: Reduced costs for nonbasic variables
        """
        # Basic objective coefficients
        c_B = self.c_extended[self.basic_indices]
        
        # Solve v^T B = c_B^T, equivalent to B^T v = c_B
        v = np.linalg.solve(self.B.T, c_B)
        
        # Compute reduced costs: c̄_j = c_j - v^T a_j for nonbasic variables
        reduced_costs = np.zeros(len(self.nonbasic_indices))
        for i, j in enumerate(self.nonbasic_indices):
            a_j = self.A_extended[:, j]
            reduced_costs[i] = self.c_extended[j] - np.dot(v, a_j)
        
        return v, reduced_costs
    
    def _check_optimality(self, reduced_costs: np.ndarray) -> Tuple[bool, Optional[int]]:
        """
        Step 2: Check optimality condition and choose entering variable.
        
        Returns:
            is_optimal: True if optimal solution found
            entering_idx: Index of entering variable (None if optimal)
        """
        # For maximization: optimal if all reduced costs ≤ 0
        improving_indices = np.where(reduced_costs > self.eps)[0]
        
        if len(improving_indices) == 0:
            print("Step 2: Optimal solution found (all reduced costs ≤ 0)")
            return True, None
        
        # Choose entering variable (most positive reduced cost)
        best_idx = improving_indices[np.argmax(reduced_costs[improving_indices])]
        entering_var = self.nonbasic_indices[best_idx]
        
        print(f"Step 2: Entering variable x_{entering_var} with reduced cost {reduced_costs[best_idx]:.6f}")
        return False, entering_var
    
    def _simplex_direction(self, entering_var: int) -> np.ndarray:
        """
        Step 3: Compute simplex direction.
        
        Parameters:
            entering_var: Index of entering variable
            
        Returns:
            direction: Simplex direction for basic variables
        """
        # Get entering column
        a_p = self.A_extended[:, entering_var]
        
        # Solve B * direction = -a_p for basic components
        direction = np.linalg.solve(self.B, -a_p)
        
        print(f"Step 3: Simplex direction computed for x_{entering_var}")
        return direction
    
    def _compute_step_size(self, direction: np.ndarray) -> Tuple[float, Optional[int], Optional[int]]:
        """
        Step 4: Compute maximum feasible step size and leaving variable.
        
        Parameters:
            direction: Simplex direction for basic variables
            
        Returns:
            step_size: Maximum feasible step size
            leaving_var: Index of leaving variable (None if unbounded)
            leaving_pos: Position of leaving variable in basis (None if unbounded)
        """
        current_basic = self.x[self.basic_indices]
        
        # Find minimum ratio for negative direction components
        ratios = []
        valid_indices = []
        
        for i, d in enumerate(direction):
            if d < -self.eps:  # Only consider negative direction components
                ratio = current_basic[i] / (-d)
                ratios.append(ratio)
                valid_indices.append(i)
        
        if len(ratios) == 0:
            print("Step 4: Problem is unbounded (no limiting constraints)")
            return float('inf'), None, None
        
        # Choose minimum ratio (leaving variable)
        min_ratio_idx = np.argmin(ratios)
        step_size = ratios[min_ratio_idx]
        leaving_pos = valid_indices[min_ratio_idx]
        leaving_var = self.basic_indices[leaving_pos]
        
        print(f"Step 4: Step size λ = {step_size:.6f}, leaving variable x_{leaving_var}")
        return step_size, leaving_var, leaving_pos
    
    def _update_solution_and_basis(self, entering_var: int, leaving_var: int, 
                                 leaving_pos: int, step_size: float, direction: np.ndarray):
        """
        Step 5: Update solution and basis representation.
        
        Parameters:
            entering_var: Index of entering variable
            leaving_var: Index of leaving variable  
            leaving_pos: Position of leaving variable in basis
            step_size: Step size
            direction: Simplex direction
        """
        # Update basic variables: x_B^(t+1) = x_B^(t) + λ * direction
        self.x[self.basic_indices] += step_size * direction
        self.x[entering_var] = step_size
        self.x[leaving_var] = 0
        
        # Update basis
        self.basic_indices[leaving_pos] = entering_var
        self.nonbasic_indices = [j for j in range(self.n_extended) if j not in self.basic_indices]
        
        # Update basis matrix and its inverse using pivot operations
        self._update_basis_inverse(entering_var, leaving_pos, direction)
        
        # Increment iteration counter
        self.t += 1
        
        print(f"Step 5: Basis updated, iteration {self.t} complete")
        print(f"New basic variables: {self.basic_indices}")
    
    def _update_basis_inverse(self, entering_var: int, leaving_pos: int, direction: np.ndarray):
        """Update B^(-1) using pivot matrix multiplication."""
        # Create pivot matrix E
        E = np.eye(self.m)
        pivot_element = -direction[leaving_pos]
        
        for i in range(self.m):
            if i != leaving_pos:
                E[i, leaving_pos] = direction[i] / pivot_element
            else:
                E[i, leaving_pos] = -1.0 / pivot_element
        
        # Update: B^(-1) = E * B^(-1) 
        self.B_inv = E @ self.B_inv
        
        # Update basis matrix
        self.B[:, leaving_pos] = self.A_extended[:, entering_var]
    
    def solve(self, max_iterations: int = 1000, verbose: bool = True) -> dict:
        """
        Solve the linear program using Revised Simplex Method.
        
        Returns:
            Dictionary containing solution information
        """
        if verbose:
            print("=== REVISED SIMPLEX METHOD ===")
            print(f"Maximizing: {' + '.join(f'{c:.3f}x_{i}' for i, c in enumerate(self.c))}")
            print("Subject to constraints and x ≥ 0\n")
        
        for iteration in range(max_iterations):
            if verbose:
                print(f"\n--- ITERATION {iteration + 1} ---")
            
            # Step 1: Pricing
            v, reduced_costs = self._pricing()
            if verbose:
                print(f"Pricing vector v = {v}")
                print(f"Reduced costs = {reduced_costs}")
            
            # Step 2: Optimality check
            is_optimal, entering_var = self._check_optimality(reduced_costs)
            if is_optimal:
                break
            
            # Step 3: Simplex direction
            direction = self._simplex_direction(entering_var)
            if verbose:
                print(f"Direction = {direction}")
            
            # Step 4: Step size
            step_size, leaving_var, leaving_pos = self._compute_step_size(direction)
            if leaving_var is None:
                return {
                    'status': 'unbounded',
                    'solution': None,
                    'objective_value': float('inf'),
                    'iterations': iteration + 1
                }
            
            # Step 5: Update
            self._update_solution_and_basis(entering_var, leaving_var, leaving_pos, 
                                          step_size, direction)
            
            if verbose:
                basic_solution = self.x[self.basic_indices]
                print(f"Updated solution: x_B = {basic_solution}")
        
        else:
            return {
                'status': 'max_iterations',
                'solution': None,
                'objective_value': None,
                'iterations': max_iterations
            }
        
        # Extract final solution
        solution = self.x[:self.n]  # Only original variables
        objective_value = np.dot(self.c, solution)
        
        return {
            'status': 'optimal',
            'solution': solution,
            'objective_value': objective_value,
            'basic_variables': self.basic_indices,
            'basic_solution': self.x[self.basic_indices],
            'iterations': self.t
        }


# Example usage and testing
if __name__ == "__main__":
    # Example 1: Standard LP problem
    # Maximize 3x₁ + 2x₂
    # Subject to: x₁ + x₂ ≤ 4
    #            2x₁ + x₂ ≤ 6
    #            x₁, x₂ ≥ 0
    
    print("EXAMPLE 1:")
    c = [3, 2]
    A = [[1, 1],
         [2, 1]]
    b = [4, 6]
    
    solver = RevisedSimplexSolver(c, A, b)
    result = solver.solve(verbose=True)
    
    print(f"\n=== FINAL RESULT ===")
    print(f"Status: {result['status']}")
    if result['status'] == 'optimal':
        print(f"Optimal solution: {result['solution']}")
        print(f"Optimal objective value: {result['objective_value']:.6f}")
        print(f"Basic variables: {result['basic_variables']}")
        print(f"Iterations: {result['iterations']}")
    
    # Example 2: Different problem for validation
    print("\n" + "="*50)
    print("EXAMPLE 2:")
    c2 = [1, 3, 2]
    A2 = [[1, 1, 1],
          [2, 1, 3]]
    b2 = [5, 8]
    
    solver2 = RevisedSimplexSolver(c2, A2, b2)
    result2 = solver2.solve(verbose=False)
    
    print(f"Status: {result2['status']}")
    if result2['status'] == 'optimal':
        print(f"Optimal solution: {result2['solution']}")
        print(f"Optimal objective value: {result2['objective_value']:.6f}")