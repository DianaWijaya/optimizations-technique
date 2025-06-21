import numpy as np
from typing import List, Tuple, Optional, Union, Dict
from enum import Enum

class VariableStatus(Enum):
    BASIC = "basic"
    NONBASIC_LOWER = "nonbasic_lower" 
    NONBASIC_UPPER = "nonbasic_upper"

class BoundedRevisedSimplex:
    def __init__(self, c: List[float], A: List[List[float]], b: List[float], 
                 lower_bounds: List[float] = None, upper_bounds: List[float] = None, 
                 eps: float = 1e-10):
        """
        Initialize Bounded Revised Simplex solver.
        
        Parameters:
            c: Objective coefficients (to maximize)
            A: Constraint matrix Ax = b
            b: Right-hand side
            lower_bounds: Lower bounds for variables (default: 0)
            upper_bounds: Upper bounds for variables (default: infinity)
            eps: Numerical tolerance
        """
        self.c = np.array(c, dtype=float)
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float)
        self.eps = eps
        
        self.m, self.n = self.A.shape  # m constraints, n variables
        
        # Set default bounds
        self.lower_bounds = np.array(lower_bounds if lower_bounds else [0.0] * self.n)
        self.upper_bounds = np.array(upper_bounds if upper_bounds else [np.inf] * self.n)
        
        self._validate_inputs()
        self._initialize()
    
    def _validate_inputs(self):
        """Validate input dimensions and bounds."""
        if len(self.c) != self.n:
            raise ValueError("Objective coefficients dimension mismatch")
        if len(self.b) != self.m:
            raise ValueError("RHS dimension mismatch")
        if len(self.lower_bounds) != self.n or len(self.upper_bounds) != self.n:
            raise ValueError("Bounds dimension mismatch")
        if np.any(self.lower_bounds > self.upper_bounds):
            raise ValueError("Lower bounds cannot exceed upper bounds")
    
    def _initialize(self):
        """Step 0: Initialize with feasible basis and variable statuses."""
        print("Step 0: Initialization")
        
        # Initialize variable statuses
        self.var_status = [VariableStatus.NONBASIC_LOWER] * self.n
        self.x = self.lower_bounds.copy()
        
        # Find initial feasible basis using artificial variables if needed
        self._find_initial_basis()
        
        # Compute initial basic solution
        self._update_basic_solution()
        
        self.t = 0  # Solution index
        
        print(f"Initial basic variables: {self.basic_vars}")
        print(f"Initial solution: x = {self.x}")
        print(f"Variable statuses: {[s.value for s in self.var_status]}")
    
    def _find_initial_basis(self):
        """Find initial feasible basis."""
        # Try to find a basis from identity columns first
        identity_cols = []
        for i in range(self.m):
            # Look for unit vectors in A
            for j in range(self.n):
                col = self.A[:, j]
                if (np.abs(col[i] - 1.0) < self.eps and 
                    np.sum(np.abs(col)) < 1 + self.eps and
                    j not in identity_cols):
                    identity_cols.append(j)
                    break
        
        if len(identity_cols) == self.m:
            # Found natural basis
            self.basic_vars = identity_cols
            for j in self.basic_vars:
                self.var_status[j] = VariableStatus.BASIC
        else:
            # Use first m variables as basis (may need Phase I in practice)
            self.basic_vars = list(range(min(self.m, self.n)))
            for j in self.basic_vars:
                self.var_status[j] = VariableStatus.BASIC
        
        # Get basis matrix and its inverse
        self.B = self.A[:, self.basic_vars]
        try:
            self.B_inv = np.linalg.inv(self.B)
        except np.linalg.LinAlgError:
            raise ValueError("Initial basis is singular")
    
    def _update_basic_solution(self):
        """Update basic variable values based on current nonbasic values."""
        # Compute RHS for basic variables: B*x_B = b - A_N*x_N
        nonbasic_contribution = np.zeros(self.m)
        
        for j in range(self.n):
            if self.var_status[j] != VariableStatus.BASIC:
                nonbasic_contribution += self.A[:, j] * self.x[j]
        
        rhs = self.b - nonbasic_contribution
        basic_solution = self.B_inv @ rhs
        
        # Update basic variables
        for i, j in enumerate(self.basic_vars):
            self.x[j] = basic_solution[i]
    
    def _pricing(self) -> np.ndarray:
        """Step 1: Compute pricing vector and reduced costs."""
        print("Step 1: Pricing")
        
        # Get basic objective coefficients
        c_B = self.c[self.basic_vars]
        
        # Solve v^T * B = c_B^T
        v = self.B_inv.T @ c_B
        
        # Compute reduced costs for all variables
        reduced_costs = self.c - self.A.T @ v
        
        print(f"Pricing vector v = {v}")
        print(f"Reduced costs = {reduced_costs}")
        
        return reduced_costs
    
    def _check_optimality(self, reduced_costs: np.ndarray) -> Tuple[bool, Optional[int], int]:
        """
        Step 2: Check optimality and choose entering variable.
        
        Rules 5.50 and 5.51:
        - For nonbasic at lower bound: can improve if c̄_j > 0 (move up)
        - For nonbasic at upper bound: can improve if c̄_j < 0 (move down)
        
        Returns:
            is_optimal: Whether current solution is optimal
            entering_var: Index of entering variable (None if optimal)
            orientation: +1 for lower bounded, -1 for upper bounded
        """
        print("Step 2: Optimality check")
        
        improving_vars = []
        
        for j in range(self.n):
            if self.var_status[j] == VariableStatus.NONBASIC_LOWER:
                if reduced_costs[j] > self.eps:  # Can improve by increasing
                    improving_vars.append((j, +1, reduced_costs[j]))
            elif self.var_status[j] == VariableStatus.NONBASIC_UPPER:
                if reduced_costs[j] < -self.eps:  # Can improve by decreasing
                    improving_vars.append((j, -1, abs(reduced_costs[j])))
        
        if not improving_vars:
            print("Optimal solution found!")
            return True, None, 0
        
        # Choose variable with best improvement (largest |reduced_cost|)
        best_var = max(improving_vars, key=lambda x: x[2])
        entering_var, orientation = best_var[0], best_var[1]
        
        bound_type = "lower" if orientation == 1 else "upper"
        print(f"Entering variable: x_{entering_var} (at {bound_type} bound)")
        print(f"Orientation d = {orientation}")
        
        return False, entering_var, orientation
    
    def _simplex_direction(self, entering_var: int, orientation: int) -> np.ndarray:
        """Step 3: Compute simplex direction."""
        print("Step 3: Simplex direction")
        
        # Get entering column
        a_p = self.A[:, entering_var]
        
        # Solve B * Δx = -d * a_p for basic components
        rhs = -orientation * a_p
        direction = self.B_inv @ rhs
        
        print(f"Direction for basic variables: {direction}")
        return direction
    
    def _compute_step_size(self, entering_var: int, orientation: int, 
                          direction: np.ndarray) -> Tuple[float, Optional[int], str]:
        """
        Step 4: Compute maximum feasible step size.
        
        Rule 5.52: Consider all constraints:
        1. Basic variables hitting their bounds
        2. Entering variable hitting its opposite bound
        
        Returns:
            step_size: Maximum feasible step
            blocking_var: Variable that blocks further movement
            block_type: Type of blocking ("basic_lower", "basic_upper", "entering")
        """
        print("Step 4: Step size computation")
        
        max_step = np.inf
        blocking_var = None
        block_type = None
        
        # Check basic variables hitting bounds
        for i, j in enumerate(self.basic_vars):
            current_val = self.x[j]
            direction_j = direction[i]
            
            if abs(direction_j) > self.eps:
                # Check lower bound
                if direction_j < 0:  # Moving toward lower bound
                    step_to_lower = (current_val - self.lower_bounds[j]) / (-direction_j)
                    if step_to_lower < max_step - self.eps:
                        max_step = step_to_lower
                        blocking_var = j
                        block_type = "basic_lower"
                
                # Check upper bound
                if direction_j > 0:  # Moving toward upper bound
                    step_to_upper = (self.upper_bounds[j] - current_val) / direction_j
                    if step_to_upper < max_step - self.eps:
                        max_step = step_to_upper
                        blocking_var = j
                        block_type = "basic_upper"
        
        # Check entering variable hitting its opposite bound
        if orientation == 1:  # Moving from lower to upper bound
            if self.upper_bounds[entering_var] < np.inf:
                step_to_upper = self.upper_bounds[entering_var] - self.x[entering_var]
                if step_to_upper < max_step - self.eps:
                    max_step = step_to_upper
                    blocking_var = entering_var
                    block_type = "entering_upper"
        else:  # Moving from upper to lower bound
            step_to_lower = self.x[entering_var] - self.lower_bounds[entering_var]
            if step_to_lower < max_step - self.eps:
                max_step = step_to_lower
                blocking_var = entering_var
                block_type = "entering_lower"
        
        if max_step >= np.inf - self.eps:
            print("Problem is unbounded!")
            return np.inf, None, "unbounded"
        
        print(f"Maximum step size λ = {max_step:.6f}")
        print(f"Blocking variable: x_{blocking_var} ({block_type})")
        
        return max_step, blocking_var, block_type
    
    def _update_solution_and_basis(self, entering_var: int, orientation: int,
                                  step_size: float, direction: np.ndarray,
                                  blocking_var: int, block_type: str):
        """Step 5: Update solution and basis."""
        print("Step 5: Update solution and basis")
        
        # Update basic variables: x_B = x_B + λ * direction
        for i, j in enumerate(self.basic_vars):
            self.x[j] += step_size * direction[i]
        
        # Update entering variable
        self.x[entering_var] += step_size * orientation
        
        # Handle different blocking scenarios
        if block_type.startswith("basic"):
            # Basic variable hits bound and leaves basis
            leaving_var = blocking_var
            leaving_pos = self.basic_vars.index(leaving_var)
            
            # Update basis
            self.basic_vars[leaving_pos] = entering_var
            self.var_status[entering_var] = VariableStatus.BASIC
            
            # Set leaving variable status based on which bound it hit
            if block_type == "basic_lower":
                self.var_status[leaving_var] = VariableStatus.NONBASIC_LOWER
                self.x[leaving_var] = self.lower_bounds[leaving_var]
            else:  # basic_upper
                self.var_status[leaving_var] = VariableStatus.NONBASIC_UPPER
                self.x[leaving_var] = self.upper_bounds[leaving_var]
            
            # Update basis matrix and inverse
            self._update_basis_inverse(leaving_pos, entering_var, direction)
            
            print(f"Basis change: x_{leaving_var} leaves, x_{entering_var} enters")
            
        elif block_type.startswith("entering"):
            # Entering variable hits its opposite bound (no basis change)
            if block_type == "entering_upper":
                self.var_status[entering_var] = VariableStatus.NONBASIC_UPPER
                self.x[entering_var] = self.upper_bounds[entering_var]
            else:  # entering_lower
                self.var_status[entering_var] = VariableStatus.NONBASIC_LOWER
                self.x[entering_var] = self.lower_bounds[entering_var]
            
            print(f"No basis change: x_{entering_var} moves to opposite bound")
        
        self.t += 1
        print(f"Iteration {self.t} complete")
        print(f"Current solution: x = {self.x}")
    
    def _update_basis_inverse(self, leaving_pos: int, entering_var: int, direction: np.ndarray):
        """Update basis inverse using pivot operations."""
        # Create pivot matrix E
        E = np.eye(self.m)
        pivot_element = direction[leaving_pos]
        
        if abs(pivot_element) < self.eps:
            raise ValueError(f"Pivot element too small: {pivot_element}")
        
        for i in range(self.m):
            if i != leaving_pos:
                E[i, leaving_pos] = -direction[i] / pivot_element
            else:
                E[i, leaving_pos] = 1.0 / pivot_element
        
        # Update B^(-1) = E * B^(-1)
        self.B_inv = E @ self.B_inv
        
        # Update basis matrix
        self.B[:, leaving_pos] = self.A[:, entering_var]
    
    def solve(self, max_iterations: int = 1000, verbose: bool = True) -> Dict:
        """Solve using Bounded Revised Simplex Method."""
        if verbose:
            print("=== BOUNDED REVISED SIMPLEX METHOD ===")
            print(f"Variables: {self.n}, Constraints: {self.m}")
            print(f"Lower bounds: {self.lower_bounds}")
            print(f"Upper bounds: {self.upper_bounds}")
            print()
        
        for iteration in range(max_iterations):
            if verbose:
                print(f"\n--- ITERATION {iteration + 1} ---")
            
            # Step 1: Pricing
            reduced_costs = self._pricing()
            
            # Step 2: Optimality
            is_optimal, entering_var, orientation = self._check_optimality(reduced_costs)
            if is_optimal:
                break
            
            # Step 3: Direction
            direction = self._simplex_direction(entering_var, orientation)
            
            # Step 4: Step size
            step_size, blocking_var, block_type = self._compute_step_size(
                entering_var, orientation, direction)
            
            if block_type == "unbounded":
                return {
                    'status': 'unbounded',
                    'solution': None,
                    'objective_value': float('inf'),
                    'iterations': iteration + 1
                }
            
            # Step 5: Update
            self._update_solution_and_basis(entering_var, orientation, 
                                          step_size, direction, 
                                          blocking_var, block_type)
        else:
            return {
                'status': 'max_iterations',
                'solution': None,
                'objective_value': None,
                'iterations': max_iterations
            }
        
        # Final solution
        objective_value = np.dot(self.c, self.x)
        
        return {
            'status': 'optimal',
            'solution': self.x.copy(),
            'objective_value': objective_value,
            'basic_variables': self.basic_vars.copy(),
            'variable_statuses': [s.value for s in self.var_status],
            'iterations': self.t
        }


# Example usage
if __name__ == "__main__":
    print("EXAMPLE 1: Standard bounds (x ≥ 0)")
    # Maximize 3x₁ + 2x₂
    # Subject to: x₁ + x₂ = 4
    #            2x₁ + x₂ = 6
    #            x₁, x₂ ≥ 0
    
    c1 = [3, 2]
    A1 = [[1, 1], [2, 1]]
    b1 = [4, 6]
    
    solver1 = BoundedRevisedSimplex(c1, A1, b1)
    result1 = solver1.solve(verbose=True)
    
    print(f"\n=== RESULT 1 ===")
    print(f"Status: {result1['status']}")
    if result1['status'] == 'optimal':
        print(f"Solution: {result1['solution']}")
        print(f"Objective: {result1['objective_value']:.6f}")
    
    print("\n" + "="*60)
    print("EXAMPLE 2: With upper bounds")
    # Maximize x₁ + 2x₂
    # Subject to: x₁ + x₂ = 3
    #            0 ≤ x₁ ≤ 2
    #            0 ≤ x₂ ≤ 2
    
    c2 = [1, 2]
    A2 = [[1, 1]]
    b2 = [3]
    lower2 = [0, 0]
    upper2 = [2, 2]
    
    solver2 = BoundedRevisedSimplex(c2, A2, b2, lower2, upper2)
    result2 = solver2.solve(verbose=False)
    
    print(f"=== RESULT 2 ===")
    print(f"Status: {result2['status']}")
    if result2['status'] == 'optimal':
        print(f"Solution: {result2['solution']}")
        print(f"Objective: {result2['objective_value']:.6f}")
        print(f"Variable statuses: {result2['variable_statuses']}")