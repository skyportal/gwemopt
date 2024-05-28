import time

import numpy as np
import pulp
from tqdm import tqdm


def solve_milp(
    cost_matrix,
    max_tasks_per_worker=1,
    useTaskSepration=False,
    min_task_separation=1,
    useDistance=False,
    dist_matrix=None,
    timeLimit=300,
    milpSolver="PULP_CBC_CMD",
    milpOptions={},
):

    cost_matrix_mask = cost_matrix > 10 ** (-10)
    optimal_points = []

    if cost_matrix_mask.any():
        print("Calculating MILP solution...")

        # Create a CP-SAT model
        problem = pulp.LpProblem("problem", pulp.LpMaximize)

        # Define variables
        num_exposures, num_fields = cost_matrix.shape

        print("Define decision variables...")
        # Define decision variables
        x = {
            (i, j): pulp.LpVariable(f"x_{i}_{j}", cat=pulp.LpBinary)
            for i in tqdm(range(num_exposures))
            for j in range(num_fields)
        }

        print("Define binary variables for task separation violation...")
        # Define binary variables for task separation violation
        s = {
            (i, j): pulp.LpVariable(f"s_{i}_{j}", cat=pulp.LpBinary)
            for i in tqdm(range(num_exposures))
            for j in range(num_fields)
        }

        obj = pulp.lpSum(
            x[i, j] * cost_matrix[i][j]
            for i in range(num_exposures)
            for j in range(num_fields)
        )

        if useDistance:
            products = [
                (x[i, j], dist_matrix[j, k])
                for i in range(num_exposures)
                for j in range(num_fields)
                for k in range(dist_matrix.shape[1])
            ]
            total_distance = pulp.LpAffineExpression(products)
            obj -= total_distance

        # One field per exposure
        for i in range(num_exposures):
            problem += pulp.lpSum(x[i, j] for j in range(num_fields)) == 1

        # Limit the number of tasks each worker can handle (if applicable)
        for j in range(num_fields):
            problem += (
                pulp.lpSum(x[i, j] for i in range(num_exposures))
                <= max_tasks_per_worker
            )

        print("Add constraints to exclude impossible assignments...")
        # Add constraints to exclude impossible assignments
        for i in tqdm(range(num_exposures)):
            for j in range(num_fields):
                if not np.isfinite(cost_matrix[i][j]):
                    problem += x[i, j] == 0

        if useTaskSepration:
            print("Define constraints: enforce minimum task separation...")
            ## Define constraints: enforce minimum task separation
            for i in tqdm(range(num_exposures)):
                for j in range(num_fields):
                    for k in range(j + 1, num_fields):
                        problem += s[i, j] >= x[i, k] - x[i, j] - min_task_separation
                        problem += s[i, j] >= x[i, j] - x[i, k] - min_task_separation

        print("Number of variables:", len(problem.variables()))
        print("Number of constraints:", len(problem.constraints))

        time_limit = 60  # Stop the solver after 60 seconds

        if milpSolver == "PULP_CBC_CMD":
            solver = pulp.getSolver(milpSolver)
        elif milpSolver == "GUROBI":
            solver = pulp.getSolver(
                "GUROBI", manageEnv=True, envOptions=milpOptions, mip_gap=1e-6
            )
        else:
            raise ValueError("milpSolver must be either PULP_CBC_CMD or GUROBI")

        solver.timeLimit = timeLimit
        # solver.msg = True
        status = problem.solve(solver)

        optimal_points = []
        if status in [pulp.LpStatusOptimal]:
            for i in range(num_exposures):
                for j in range(num_fields):
                    if (
                        pulp.value(x[i, j]) is not None
                        and pulp.value(x[i, j]) > 0.5
                        and np.isfinite(cost_matrix[i][j])
                    ):
                        optimal_points.append((i, j))
        else:
            print("The problem does not have a solution.")

    else:
        print("The localization is not visible from the site.")

    return optimal_points
