"""
The numerov module handles the routines required to solve the KS equations.

So far, there is only a single implementation which is based on a matrix
diagonalization. There is an option for parallelizing over the angular
quantum number `l`.

Functions
---------
* :func:`matrix_solve` : Solve the radial KS equation via matrix diagonalization of \
                         Numerov's method.
* :func:`KS_matsolve_parallel` : Solve the KS matrix diagonalization by parallelizing \
                                 over config.ncores.
* :func:`KS_matsolve_serial` : Solve the KS matrix diagonalization in serial.
* :func:`diag_H` : Diagonalize the Hamiltonian for given input potential.
* :func:`update_orbs` : Sort the eigenvalues and functions by ascending energies and \
                        normalize orbs.
"""


# standard libs
import os
import shutil

# external libs
import numpy as np
from scipy.sparse.linalg import eigs
from joblib import Parallel, delayed, dump, load


#ZZZ
import math as ma

# from staticKS import Orbitals

# internal libs
from . import config
from . import mathtools

def ShootGen(V, xgrid):   #The main body of the shooting method solution. calls different functions within the Numerov.py module. Returns the eigenvalues and functions.
    eigvals=np.zeros(config.spindims, config.lmax, config.nmax)
    eigfuncs=np.zeros(config.spindims, comfig.lmax, config.nmax, config.grig_params["ngrid"])
    max_Delta=pow(10,-3) #Will be optimized
    flag=bool(False)
    for l in range(config.lmax):
        for n in range(config.nmax):
            if (l>=n):
                break
            else:
            #Obtain (Somehow) an initial E_r value
                Z_r=Shootsolve(v, xgrid, l, E_r)
                E_l=E_r
                Delta=0.1
                while (flag==False):
                    E_l=E_l-Delta
                    Z_l=Shootsolve(v, xgrid, l, E_l)
                    if (Z_l*Z_r<=0):
                        E_l=E_l+Delta
                        Delta =0.5*Delta
                        if (Delta<max_Delta): #ZZZ Check if there's user defined max_delta
                            flag=bool(True)
                            break
                    else:
                        continue
            eigvals(0,l,n)=E_l
            R=Shootwrite(v, xgrid, l, E)
            while i in range(xgrid):
                eigfuncs(0,l,n,i)=R[i]
            n=+1
        l=+1
    return eigfuncs, eigvals

def Shootsolve(v, xgrid, l, E): #Solves the KS equation by the shooting method for the P_nl. It dosn't write down anything but the value of the derivative continuation.
    #Defining spatial resolution -dx, N-No. of points, k-"Numerov's" k(x)- defined as a zeros array, W- k in the log scale
    #u and y are for performing the integration
    N = int(np.size(xgrid))
    dx = xgrid[1] - xgrid[0] 
    h=dx**2
    k=v-E #ZZZ
    W=np.zeros(N)
    u=np.zeros(3)
    y=np.zeros(3)
    tp=0

    #(i) Searching for turning point:
    for i in range(xgrid):
        W[i]=-2.0*ma.exp(2.0*x[i])*(v[i]+0.5*(l+0.5)**2*ma.exp(-2.0*x[i])-E) #Defining the logarithmic k array
        if ((tp=0) and (k[i]*k[i-1]<0) and (abs(k[i]-k[i-1])<1.0)): #Searching for a turningpoint
            tp=int(i-1)
        i=+1
    #(ii) Forward integration        
    a=config.grid_params["x0"]  #a is the leftmost grid point. 
    u[0]=a
    u[1]=ma.exp((l+0.5)*dx)*a
    i=int(1)
    while (i<tp): #

        u[2]=(2.0*(1.0-5.0*h/12.0*W[i])*u[1]-(1.0+h/12.0*W[i-1])*u[0])/(1.0+h/12.0*W[i+1])
            
        u[0]=u[1]
        u[1]=u[2]
        i=+1
        
    y[0]=u[0] 
    y[1]=u[1]

    y[2]=(2.0*(1.0-5.0*h/12.0*W[tp-1])*y[1]-(1.0+h/12.0*W[tp-2])*y[0])/(1.0+h/12.0*W[tp])
        
    lef=-(y[3]-y[1])/y[2]
        
    #(iii) Backwards integration
    u[0]=u[1]=u[2]=0.0
        
    if (config.bc=="neumann"):
        u[2]=u[1]=0 #ZZZ Const?
    else:
        u[2]=0
        u[1]=a #ZZZ

    i=int(np.size(xgrid)-1)
    while (i>tp):
        u[0]=(-(1.0+h/12.0*W[i+1])*u[2]+2.0*(1.0-5.0*h/12.0*W[i]*u[1]))/(1.0+h/12.0*W[i-1])

        u[2]=u[1]
        u[1]=u[0]
        i=-1
        
    y[2]=u[1]
    y[1]=u[0]

    y[0]=(-(1.0+h/12*W[tp+2])*y[2]+2.0*(1.0-5.0*h/12.0*W[tp+1])*y[1])/(1.0+h/12.0*W[tp])
        
    right=(y[1]-y[3])/y[2]

    #Defining the "differntiability" function for the specific (E,l)
    cont=lef+right
    return cont
            
def Shootwrite(v, xgrid, l, E): #Solves the KS equation for P_nl, writes it down and reports it. 
    dx = xgrid[1] - xgrid[0]
    N = int(np.size(xgrid))
    h=dx**2
    k=v-E #ZZZ
    W=np.zeros(N)
    P=np.zeros(N)
    R=np.zeros(N)

    #(i) Searching for turning point:
    for i in range(xgrid):
        W[i]=-2.0*ma.exp(2.0*x[i])*(v[i]+0.5*(l+0.5)**2*ma.exp(-2.0*x[i])-E) #Defining the logarithmic k array
        if ((tp=0) and (k[i]*k[i-1]<0) and (abs(k[i]-k[i-1])<1.0)):
            tp=int(i-1)
        i=+1   

    #(ii) Forward integration        
    a=config.grid_params["x0"]  #a is the leftmost grid point. 
    P[0]=a
    P[1]=ma.exp((l+0.5)*dx)*a
    i=int(2)
    while (i<tp): 

        P[i]=(2.0*(1.0-5.0*h/12.0*W[i-1])*P[i-1]-(1.0+h/12.0*W[i-2])*P[i-2])/(1.0+h/12.0*W[i])
        i=+1
        

    P[tp]=(2.0*(1.0-5.0*h/12.0*W[tp-1])*P[tp-1]-(1.0+h/12.0*W[tp-2])*P[tp-2])/(1.0+h/12.0*W[tp])
    check=P[tp]

    #(iii) Backwards integration
        
    if (config.bc=="neumann"):
        P[N-1]=P[N-2]=0 #ZZZ Const?
    else:
        P[N-1]=0
        P[N-2]=a #ZZZ

    i=int(np.size(xgrid)-3)
    while (i>=tp):
        P[i]=(-(1.0+h/12.0*W[i+2])*P[i+2]+2.0*(1.0-5.0*h/12.0*W[i+1]*P[i+1]))/(1.0+h/12.0*W[i])
        i=-1

    #ZZZ Will we get function continuity?
    return P


# @writeoutput.timing
def matrix_solve(v, xgrid):
    r"""
    Solve the radial KS equation via matrix diagonalization of Numerov's method.

    See notes for details of the implementation.

    Parameters
    ----------
    v : ndarray
        the KS potential on the log grid
    xgrid : ndarray
        the logarithmic grid

    Returns
    -------
    eigfuncs : ndarray
        the radial KS eigenfunctions on the log grid
    eigvals : ndarray
        the KS eigenvalues

    Notes
    -----
    The implementation is based on [2]_.

    The matrix diagonalization is of the form:

    .. math::

        \hat{H} \lvert X \rangle &= \lambda \hat{B} \lvert X \rangle\ , \\
        \hat{H}                  &= \hat{T} + \hat{B}\times\hat{V}\ ,   \\
        \hat{T}                  &= -0.5\times\hat{p}\times\hat{A}\ .

    where :math:`\hat{p}=\exp(-2x)`.
    See [2]_ for the definitions of the matrices :math:`\hat{A}` and :math:`\hat{B}`.

    References
    ----------
    .. [2] M. Pillai, J. Goglio, and T. G. Walker , Matrix Numerov method for solving
       Schrödinger’s equation, American Journal of Physics 80,
       1017-1019 (2012) `DOI:10.1119/1.4748813 <https://doi.org/10.1119/1.4748813>`__.
    """
    N = config.grid_params["ngrid"]

    # define the spacing of the xgrid
    dx = xgrid[1] - xgrid[0]
    # number of grid points

    # Set-up the following matrix diagonalization problem
    # H*|u>=E*B*|u>; H=T+B*V; T=-p*A
    # |u> is related to the radial eigenfunctions R(r) via R(x)=exp(x/2)u(x)

    # off-diagonal matrices
    I_minus = np.eye(N, k=-1)
    I_zero = np.eye(N)
    I_plus = np.eye(N, k=1)

    p = np.zeros((N, N))  # transformation for kinetic term on log grid
    np.fill_diagonal(p, np.exp(-2 * xgrid))

    # see referenced paper for definitions of A and B matrices
    A = np.matrix((I_minus - 2 * I_zero + I_plus) / dx ** 2)
    B = np.matrix((I_minus + 10 * I_zero + I_plus) / 12)

    # von neumann boundary conditions
    if config.bc == "neumann":
        A[N - 2, N - 1] = 2 * dx ** (-2)
        B[N - 2, N - 1] = 2 * B[N - 2, N - 1]
        A[N - 1, N - 1] = A[N - 1, N - 1] + 1.0 / dx
        B[N - 1, N - 1] = B[N - 1, N - 1] - dx / 12.0

    # construct kinetic energy matrix
    T = -0.5 * p * A

    # solve in serial or parallel - serial mostly useful for debugging
    if config.numcores > 0:
        eigfuncs, eigvals = KS_matsolve_parallel(T, B, v, xgrid)
    else:
        eigfuncs, eigvals = KS_matsolve_serial(T, B, v, xgrid)

    return eigfuncs, eigvals


def KS_matsolve_parallel(T, B, v, xgrid):
    """
    Solve the KS matrix diagonalization by parallelizing over config.ncores.

    Parameters
    ----------
    T : ndarray
        kinetic energy array
    B : ndarray
        off-diagonal array (for RHS of eigenvalue problem)
    v : ndarray
        KS potential array
    xgrid : ndarray
        the logarithmic grid

    Returns
    -------
    eigfuncs : ndarray
        radial KS wfns
    eigvals : ndarray
        KS eigenvalues
    """
    # compute the number of grid points
    N = np.size(xgrid)

    # Compute the number pmax of distinct diagonizations to be solved
    pmax = config.spindims * config.lmax

    # now flatten the potential matrix over spins
    v_flat = np.zeros((pmax, N))
    for i in range(np.shape(v)[0]):
        for l in range(config.lmax):
            v_flat[l + (i * config.lmax)] = v[i] + 0.5 * (l + 0.5) ** 2 * np.exp(
                -2 * xgrid
            )

    # make temporary folder to store arrays
    joblib_folder = "./joblib_memmap"
    try:
        os.mkdir(joblib_folder)
    except FileExistsError:
        pass

    # dump and load the large numpy arrays from file
    data_filename_memmap = os.path.join(joblib_folder, "data_memmap")
    dump((T, B, v_flat), data_filename_memmap)
    T, B, v_flat = load(data_filename_memmap, mmap_mode="r")

    # set up the parallel job
    with Parallel(n_jobs=config.numcores) as parallel:
        X = parallel(
            delayed(diag_H)(q, T, B, v_flat, xgrid, config.nmax, config.bc)
            for q in range(pmax)
        )

    # remove the joblib arrays
    try:
        shutil.rmtree(joblib_folder)
    except:  # noqa
        print("Could not clean-up automatically.")

    # retrieve the eigfuncs and eigvals from the joblib output
    eigfuncs_flat = np.zeros((pmax, config.nmax, N))
    eigvals_flat = np.zeros((pmax, config.nmax))
    for q in range(pmax):
        eigfuncs_flat[q] = X[q][0]
        eigvals_flat[q] = X[q][1]

    # unflatten eigfuncs / eigvals so they return to original shape
    eigfuncs = eigfuncs_flat.reshape(config.spindims, config.lmax, config.nmax, N)
    eigvals = eigvals_flat.reshape(config.spindims, config.lmax, config.nmax)

    return eigfuncs, eigvals


def KS_matsolve_serial(T, B, v, xgrid):
    """
    Solve the KS equations via matrix diagonalization in serial.

    Parameters
    ----------
    T : ndarray
        kinetic energy array
    B : ndarray
        off-diagonal array (for RHS of eigenvalue problem)
    v : ndarray
        KS potential array
    xgrid : ndarray
        the logarithmic grid

    Returns
    -------
    eigfuncs : ndarray
        radial KS wfns
    eigvals : ndarray
        KS eigenvalues
    """
    # compute the number of grid points
    N = np.size(xgrid)
    # initialize empty potential matrix
    V_mat = np.zeros((N, N))

    # initialize the eigenfunctions and their eigenvalues
    eigfuncs = np.zeros((config.spindims, config.lmax, config.nmax, N))
    eigvals = np.zeros((config.spindims, config.lmax, config.nmax))

    # A new Hamiltonian has to be re-constructed for every value of l and each spin
    # channel if spin-polarized
    for l in range(config.lmax):

        # diagonalize Hamiltonian using scipy
        for i in range(np.shape(v)[0]):

            # fill potential matrices
            np.fill_diagonal(V_mat, v[i] + 0.5 * (l + 0.5) ** 2 * np.exp(-2 * xgrid))

            # construct Hamiltonians
            H = T + B * V_mat

            # we seek the lowest nmax eigenvalues from sparse matrix diagonalization
            # use `shift-invert mode' (sigma=0) and pick lowest magnitude ("LM") eigs
            # sigma=0 seems to cause numerical issues so use a small offset
            eigs_up, vecs_up = eigs(H, k=config.nmax, M=B, which="LM", sigma=0.0001)

            eigfuncs[i, l], eigvals[i, l] = update_orbs(
                vecs_up, eigs_up, xgrid, config.bc
            )

    return eigfuncs, eigvals


def diag_H(p, T, B, v, xgrid, nmax, bc):
    """
    Diagonilize the Hamiltonian for the input potential v[p].

    Uses Scipy's sparse matrix solver scipy.sparse.linalg.eigs. This
    searches for the lowest magnitude `nmax` eigenvalues, so care
    must be taken to converge calculations wrt `nmax`.

    Parameters
    ----------
    p : int
       the desired index of the input array v to solve for
    T : ndarray
        the kinetic energy matrix
    B : ndarray
        the off diagonal matrix multiplying V and RHS
    v : ndarray
        KS potential array
    xgrid : ndarray
        the logarithmic grid
    nmax : int
        number of eigenvalues returned by the sparse matrix diagonalization
    bc : str
        the boundary condition

    Returns
    -------
    evecs : ndarray
        the KS radial eigenfunctions
    evals : ndarray
        the KS eigenvalues
    """
    # compute the number of grid points
    N = np.size(xgrid)
    # initialize empty potential matrix
    V_mat = np.zeros((N, N))

    # fill potential matrices
    # np.fill_diagonal(V_mat, v + 0.5 * (l + 0.5) ** 2 * np.exp(-2 * xgrid))
    np.fill_diagonal(V_mat, v[p])

    # construct Hamiltonians
    H = T + B * V_mat

    # we seek the lowest nmax eigenvalues from sparse matrix diagonalization
    # use `shift-invert mode' (sigma=0) and pick lowest magnitude ("LM") eigs
    # sigma=0 seems to cause numerical issues so use a small offset
    evals, evecs = eigs(H, k=nmax, M=B, which="LM", sigma=0.0001)

    # sort and normalize
    evecs, evals = update_orbs(evecs, evals, xgrid, bc)

    return evecs, evals


def update_orbs(l_eigfuncs, l_eigvals, xgrid, bc):
    """
    Sort the eigenvalues and functions by ascending energies and normalize orbs.

    Parameters
    ----------
    l_eigfuncs : ndarray
        input (unsorted and unnormalized) eigenfunctions (for given l and spin)
    l_eigvals : ndarray
        input (unsorted) eigenvalues (for given l and spin)
    xgrid : ndarray
        the logarithmic grid
    bc : str
        the boundary condition

    Returns
    -------
    eigfuncs : ndarray
        sorted and normalized eigenfunctions
    eigvals : ndarray
        sorted eigenvalues in ascending energy
    """
    # Sort eigenvalues in ascending order
    idr = np.argsort(l_eigvals)
    eigvals = np.array(l_eigvals[idr].real)
    # under neumann bc the RHS pt is junk, convert to correct value
    if bc == "neumann":
        l_eigfuncs[-1] = l_eigfuncs[-2]
    eigfuncs = np.array(np.transpose(l_eigfuncs.real)[idr])
    eigfuncs = mathtools.normalize_orbs(eigfuncs, xgrid)  # normalize

    return eigfuncs, eigvals
