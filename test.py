import matplotlib.pyplot as plt
from solver import solve_problem

use_case = 'OAF'

num_eigen = 2000
t = 6000

params = {}

params['OAF'] = {
    'k'     : 5,
    'm'     : 7,
    'nu'    : 3.14,
    'C'     : 9.,
    'x0'    : 5,
    'xq'    : 3,
    'dt'    : 0.5,
    'steps' : 100
    }

params['OAA'] = {
    'k'     : 5,
    'm'     : 7,
    'nu'    : 0.4,
    'C'     : 9,
    'x0'    : 5,
    'xq'    : 2,
    'dt'    : 0.5,
    'steps' : 100,
    'gamma' : 0.1
    }

params['C2D'] = {
    'k'     : 3,
    'u1x'    : 5,
    'u2x'    : 3,
    'u1y'   : 4,
    'u2y'   : 2,
    'dxy'    : 0.5,
    'nx'    : 20,
    'ny'    : 20
    }

# -----------------------------------------------------

# Test all problem types

algorithm_result, actual_result, x_axis, result_2d = solve_problem(
    problem=use_case,
    params=params[use_case], 
    num_eigen=num_eigen,
    t=t
)
if use_case in ['OAF', 'OAA']:
    plt.figure('Forced oscillator', figsize=(10,6))
    plt.plot(x_axis, actual_result, 'b-', linewidth=3, label='PyTorch')
    plt.plot(x_axis, algorithm_result, 'r.', markersize=10, label='TN')

    plt.xlabel('t'); plt.ylabel('x')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

elif use_case == 'C2D':
    # Plot 1D comparison
    plt.figure('C2Heat', figsize=(10, 6))
    plt.plot(x_axis, actual_result, 'b-', linewidth=2, label='PyTorch')
    plt.plot(x_axis, algorithm_result, 'r.', markersize=10, label='TN')

    plt.xlabel('(x, y)'); plt.ylabel('T(x,y)')
    plt.legend(loc='upper right'); plt.tight_layout()
    plt.savefig('C2D.pdf')
    plt.show()

    # Plot 2D heatmap
    plt.figure('Heat equation', figsize=(10, 6))
    plt.pcolormesh(result_2d, cmap="CMRmap")
    plt.colorbar()
    plt.xlabel('x'); plt.ylabel('y')
    plt.tight_layout()
    plt.show()
