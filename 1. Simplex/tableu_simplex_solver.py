import numpy as np

class SimplexSolver:
    def __init__(self, c, A, b, eps=1e-10):
        """
        Initialize the Simplex solver.
        Parameters:
            c (List[float]): Coefficients of the objective function (to maximize)
            A (List[List[float]]): Constraint matrix (each row: ≤ constraint)
            b (List[float]): Right-hand side of constraints
            eps (float): Tolerance for numerical comparisons
        """
        self.c = np.array(c, dtype=float)
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float)
        self.eps = eps
        
        # Validation
        if len(self.A.shape) != 2:
            raise ValueError("A must be a 2D matrix")
        if self.A.shape[0] != len(self.b):
            raise ValueError("Number of constraints in A and b must match")
        if self.A.shape[1] != len(self.c):
            raise ValueError("Number of variables in A and c must match")
        if np.any(self.b < 0):
            raise ValueError("All constraints must have non-negative RHS (b >= 0)")
        
        self.num_vars = len(c)
        self.num_constraints = len(b)
        self._create_tableau()
    
    def _create_tableau(self):
        """Create the initial simplex tableau with slack variables."""
        # Add slack variables (identity matrix)
        identity = np.eye(self.num_constraints)
        self.tableau = np.hstack((self.A, identity, self.b.reshape(-1, 1)))
        
        # Add objective row (negate c for maximization)
        obj_row = np.hstack((-self.c, np.zeros(self.num_constraints + 1)))
        self.tableau = np.vstack((self.tableau, obj_row))
        
        # Track basic variables (initially slack variables)
        self.basic_vars = list(range(self.num_vars, self.num_vars + self.num_constraints))
    
    def _pivot(self, row, col):
        """Perform a pivot operation on tableau[row][col]."""
        pivot_element = self.tableau[row][col]
        if abs(pivot_element) < self.eps:
            raise ValueError(f"Pivot element too small: {pivot_element}")
        
        # Scale pivot row
        self.tableau[row] = self.tableau[row] / pivot_element
        
        # Eliminate other rows
        for r in range(len(self.tableau)):
            if r != row and abs(self.tableau[r][col]) > self.eps:
                self.tableau[r] -= self.tableau[r][col] * self.tableau[row]
        
        # Update basic variables
        self.basic_vars[row] = col
    
    def _choose_entering_variable(self):
        """Choose entering variable using the most negative coefficient in objective row."""
        last_row = self.tableau[-1, :-1]  # Exclude RHS column
        min_idx = np.argmin(last_row)
        
        # Check if optimal (all coefficients non-negative)
        if last_row[min_idx] >= -self.eps:
            return None
        return min_idx
    
    def _choose_leaving_variable(self, entering_col):
        """Choose leaving variable using the minimum ratio test."""
        ratios = []
        valid_ratios = []
        
        for i in range(self.num_constraints):
            col_value = self.tableau[i][entering_col]
            rhs = self.tableau[i][-1]
            
            if col_value > self.eps:  # Only positive denominators
                ratio = rhs / col_value
                ratios.append(ratio)
                valid_ratios.append((ratio, i))
            else:
                ratios.append(np.inf)
        
        if not valid_ratios:
            raise Exception("Unbounded solution: no positive pivot column elements")
        
        # Find minimum ratio (with tie-breaking)
        min_ratio = min(valid_ratios)[0]
        candidates = [idx for ratio, idx in valid_ratios if abs(ratio - min_ratio) < self.eps]
        
        return candidates[0]  # Simple tie-breaking: choose first
    
    def solve(self, max_iterations=1000, verbose=False):
        """
        Run the Simplex algorithm and return the optimal solution and value.
        
        Returns:
            tuple: (solution_vector, optimal_value, status)
                status: 'optimal', 'unbounded', or 'max_iterations'
        """
        iteration = 0
        
        if verbose:
            print("Initial tableau:")
            print(self.tableau)
        
        while iteration < max_iterations:
            # Choose entering variable
            entering = self._choose_entering_variable()
            if entering is None:
                if verbose:
                    print("\nOptimal solution reached.")
                break

            # Choose leaving variable
            try:
                leaving = self._choose_leaving_variable(entering)
            except Exception:
                if verbose:
                    print("\nUnbounded solution detected.")
                return None, None, 'unbounded'

            # Step 3: Verbose output
            if verbose:
                print(f"\n--- Iteration {iteration + 1} ---")
                print(f"Entering variable: x{entering}")
                print(f"Leaving variable: row {leaving} (was x{self.basic_vars[leaving]})")

            # Step 4: Pivot
            self._pivot(leaving, entering)

            if verbose:
                print("Updated tableau:")
                print(self.tableau)

            iteration += 1

        
        if iteration >= max_iterations:
            return None, None, 'max_iterations'
        
        # Extract solution
        solution = np.zeros(self.num_vars)
        for i in range(self.num_constraints):
            if self.basic_vars[i] < self.num_vars:
                solution[self.basic_vars[i]] = max(0, self.tableau[i][-1])  # Ensure non-negative
        
        optimal_value = self.tableau[-1][-1]
        
        return solution, optimal_value, 'optimal'
    
    def get_tableau(self):
        """Return current tableau for inspection."""
        return self.tableau.copy()
    
    def get_basic_variables(self):
        """Return current basic variables."""
        return self.basic_vars.copy()


# Example usage and testing
if __name__ == "__main__":
    # Example: Maximize 3x1 + 2x2
    # Subject to: x1 + x2 <= 4
    #            2x1 + x2 <= 6  
    #            x1, x2 >= 0
    
    c = [3, 2]  # Objective coefficients
    A = [[1, 1],    # Constraint matrix
         [2, 1]]
    b = [4, 6]  # RHS values
    
    solver = SimplexSolver(c, A, b)
    solution, value, status = solver.solve(verbose=True)
    
    print(f"\nFinal Result:")
    print(f"Status: {status}")
    if status == 'optimal':
        print(f"Optimal solution: {solution}")
        print(f"Optimal value: {value}")
        print(f"Basic variables: {solver.get_basic_variables()}")
        
    print("\n" + "="*50)
        
    c2 = [1, 3, 2]
    A2 = [[1, 1, 1],
          [2, 1, 3]]
    b2 = [5, 8]
    
    solver2 = SimplexSolver(c2, A2, b2)
    solution2, value2, status2 = solver2.solve(verbose=True)
    
    print(f"\nFinal Result for second problem:")
    print(f"Status: {status2}")
    if status2 == 'optimal':
        print(f"Optimal solution: {solution2}")
        print(f"Optimal value: {value2}")
        print(f"Basic variables: {solver2.get_basic_variables()}")
        