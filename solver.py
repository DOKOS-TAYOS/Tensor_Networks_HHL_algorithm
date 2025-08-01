import torch
import numpy as np

from problems import problem_OAF, problem_OAA, problem_C2D
from tn_hhl import tensornetwork_HHL

def solve_problem(problem: str, params: dict, num_eigen: int, t: float):
    """Solves different types of problems using tensor network HHL algorithm.
    
    Args:
        problem: Type of problem ('OAF', 'OAA' or 'C2D')
        params: Dictionary containing problem parameters
        num_eigen: Number of eigenvalues to use
        t: Time parameter for the algorithm
        
    Returns:
        algorithm_result: Solution from tensor network HHL
        actual_result: Solution from direct matrix inversion
        x_axis: x-axis values for plotting
        result_2d: Reshaped 2D result for C2D problem, None otherwise
    """
    # Get problem matrices and vectors
    if problem == 'OAF':
        force, matrix = problem_OAF(params, scaling=True)
    elif problem == 'OAA':
        force, matrix, old_force, old_matrix = problem_OAA(params, scaling=True)
    elif problem == 'C2D':
        force, matrix = problem_C2D(params, scaling=False)
    else:
        raise ValueError(f"Unknown problem type: {problem}")

    # Solve using tensor network HHL
    if problem in ['OAF', 'OAA']:
        if problem == 'OAF':
            algorithm_result = tensornetwork_HHL(num_eigen, t, force, matrix) * params['dt']**2
            actual_result = (torch.linalg.inv(matrix) @ force).real * params['dt']**2
        else:  # OAA
            algorithm_result = tensornetwork_HHL(num_eigen, t, force, matrix)[params['steps']:] * params['dt']**2
            actual_result = torch.matmul(torch.linalg.inv(old_matrix.real), old_force.real) * params['dt']**2

        x_axis = np.arange(params['steps']) * params['dt']
            
        return algorithm_result, actual_result, x_axis, None
        
    else:  # C2D
        algorithm_result = tensornetwork_HHL(num_eigen, t, force, matrix)
        actual_result = torch.linalg.inv(matrix.real) @ force.real
        x_axis = np.arange(params['nx'] * params['ny']) * params['dxy']

        # Reshape result to 2D and add boundary conditions
        result_2d = list(algorithm_result.reshape(params['nx'], params['ny']))
        result_2d = [[params['u1x']] * (params['nx'] + 2)] + result_2d + [[params['u2x']] * (params['nx'] + 2)]
        
        for i in range(1, params['nx'] + 1):
            result_2d[i] = [params['u1y']] + list(result_2d[i]) + [params['u2y']]
            
        return algorithm_result, actual_result, x_axis, result_2d

