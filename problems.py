import torch

import numpy as np


def problem_OAF(param, scaling = True):
    '''
    Function that creates matrix A and vector b for the forced harmonic oscillator.
    Params:
    k = spring constant
    m = mass of the spring
    nu = force frequency
    C = external force intensity
    x0 = left boundary
    xq = right boundary
    dt = time interval
    steps = number of time intervals

    - scaling, bool: If True, rescale the problem so that the matrix has norm 1.

    Output:
    - force, torch.Tensor: vector b
    - matrix, torch.Tensor: matrix A'''

    k, m, nu, C = param['k'], param['m'], param['nu'], param['C']
    x0, xq, dt, steps = param['x0'], param['xq'], param['dt'], param['steps']

    # Matrix
    matrix = torch.eye(steps, dtype=torch.complex128) * (-2 + k/m * dt**2)  # Diagonal
    for i in range(steps-1):  # Non-diagonal
        matrix[i,i+1] = 1  # Upper diagonal
        matrix[i+1,i] = 1  # Lower diagonal

    # External force
    force = torch.tensor( [C*np.sin(np.pi*nu*dt)-x0/dt**2] + \
                    [ C*np.sin(np.pi*nu*(i+1)*dt) for i in range(1, steps-1)] + \
                    [C*np.sin(np.pi*nu*steps*dt)-xq/dt**2], dtype=torch.complex128)

    if scaling == True:
        scale = torch.linalg.norm(matrix)
        matrix /= scale
        force /= scale

    return force, matrix



def problem_OAA(param, scaling = True):
    '''
    Function that creates matrix A and vector b for the forced and damped harmonic oscillator.
    Params:
    k = spring constant
    m = mass of the spring
    nu = force frequency
    gamma = damping
    C = external force intensity
    x0 = left boundary
    xq = right boundary
    dt = time interval
    steps = number of time intervals

    - scaling, bool: If True, rescale the problem so that the matrix has norm 1.'''

    k, m, nu, C, gamma = param['k'], param['m'], param['nu'], param['C'], param['gamma']
    x0, xq, dt, steps = param['x0'], param['xq'], param['dt'], param['steps']

    # Matrix
    matrix = torch.eye(steps, dtype=torch.complex128) * (-2 + k/m * dt**2)  # Diagonal
    for i in range(steps-1):  # Non-diagonal
        matrix[i,i+1] = 1+gamma*dt/2
        matrix[i+1,i] = 1-gamma*dt/2

    # External force
    force = torch.tensor( [C*np.sin(np.pi*nu*dt)-(1-gamma*dt/2)*x0/dt**2] + \
                    [ C*np.sin(np.pi*nu*(i+1)*dt) for i in range(1, steps-1)] + \
                    [C*np.sin(np.pi*nu*steps*dt)-(1+gamma*dt/2)*xq/dt**2] )

    if scaling == True:
        scale = torch.linalg.norm(matrix)
        matrix /= scale
        force /= scale

    new_matrix = torch.zeros((2*steps, 2*steps), dtype=torch.complex128)
    new_matrix[:steps, steps:] = matrix
    new_matrix[steps:, :steps] = matrix.T

    new_force = torch.zeros(2*steps, dtype=torch.complex128)
    new_force[:steps] = force
    return new_force, new_matrix, force, matrix



def problem_C2D(param, scaling = True):
    '''
    Function that creates matrix A and vector b for the 2D convection-diffusion problem.
    Params:
    k = diffusion coefficient
    u1x, u2x = boundary conditions in x direction
    u1y, u2y = boundary conditions in y direction
    dxy = spatial discretization step
    nx, ny = number of grid points in x and y directions

    - scaling, bool: If True, rescale the problem so that the matrix has norm 1.'''

    k, u1x, u2x, u1y, u2y = param['k'], param['u1x'], param['u2x'], param['u1y'], param['u2y']
    dxy, nx, ny = param['dxy'], param['nx'], param['ny']

    # Matrix
    matrix = torch.eye(nx*ny, dtype=torch.complex128) * (-4)  # Diagonal
    
    # First row (i=0)
    matrix[0,ny] = 1
    matrix[0,1] = 1
    for j in range(1, ny-1):  # Non-diagonal
        matrix[j,ny+j] = 1
        matrix[j,j+1] = 1
        matrix[j,j-1] = 1
    matrix[ny-1,ny+ny-1] = 1
    matrix[ny-1,ny-1-1] = 1

    for i in range(1, nx-1):
        # First column (j=0)
        matrix[i*ny,(i+1)*ny] = 1
        matrix[i*ny,(i-1)*ny] = 1
        matrix[i*ny,i*ny+1] = 1
        for j in range(1, ny-1):  # Non-diagonal
            matrix[i*ny+j,(i+1)*ny+j] = 1
            matrix[i*ny+j,(i-1)*ny+j] = 1
            matrix[i*ny+j,i*ny+j+1] = 1
            matrix[i*ny+j,i*ny+j-1] = 1
        # Last column (j=ny-1)
        matrix[i*ny+ny-1,(i+1)*ny+ny-1] = 1
        matrix[i*ny+ny-1,(i-1)*ny+ny-1] = 1
        matrix[i*ny+ny-1,i*ny+ny-1-1] = 1

    # Last row (i=nx-1)
    matrix[(nx-1)*ny,((nx-1)-1)*ny] = 1
    matrix[(nx-1)*ny,(nx-1)*ny+1] = 1
    for j in range(1, ny-1):  # Non-diagonal
        matrix[(nx-1)*ny+j,((nx-1)-1)*ny+j] = 1
        matrix[(nx-1)*ny+j,(nx-1)*ny+j+1] = 1
        matrix[(nx-1)*ny+j,(nx-1)*ny+j-1] = 1
    matrix[(nx-1)*ny+ny-1,((nx-1)-1)*ny+ny-1] = 1
    matrix[(nx-1)*ny+ny-1,(nx-1)*ny+ny-1-1] = 1

    # External force
    fuerza = torch.zeros((nx, ny), dtype=torch.complex128)
    fuerza[0,0] = u1x*k/dxy**2 + u1y*k/dxy**2
    for j in range(1, ny-1):
        fuerza[0,j] = u1x*k/dxy**2
    fuerza[0, ny-1] = u1x*k/dxy**2 + u2y*k/dxy**2

    for i in range(1, nx-1):
        fuerza[i,0] = u1y*k/dxy**2
        for j in range(1, ny-1):
            fuerza[i,j] = 10*np.sin(2*np.pi*i*j/np.sqrt(nx*ny))
        fuerza[i, ny-1] = u2y*k/dxy**2

    fuerza[nx-1,0] = u2x*k/dxy**2 + u1y*k/dxy**2
    for j in range(1, ny-1):
        fuerza[nx-1,j] = u2x*k/dxy**2
    fuerza[nx-1, ny-1] = u2x*k/dxy**2 + u2y*k/dxy**2

    fuerza = -fuerza.flatten()*dxy**2/k

    if scaling == True:
        scalado = torch.linalg.norm(matrix)
        matrix /= scalado
        fuerza /= scalado

    return fuerza, matrix