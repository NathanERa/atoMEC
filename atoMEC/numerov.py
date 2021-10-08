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

def Eig_shoot_search(v, xgrid):
    guess=0.1 #Initial 1/E guess
    dx=xgrid[1]-xgrid[0] #Spatial resolution
    E_max_err=10**(-5) #maximal energy error
    h4=dx**4 

    max_count=1000 #maximal number of search steps

    st=0.1 #1/E energy step
    
    eigvals=np.zeros((config.spindims, config.lmax, config.nmax))
    eigfuncs=np.zeros((config.spindims, config.lmax, config.nmax, config.grid_params["ngrid"]))
    n=int(1)
    l=int(0)
    #f=open("z.txt", "w")
    j=0
    D=Shootwrite(v, xgrid, 0, -0.502)
    
    #while j < np.size(xgrid):
    #    vstr=str(D[j])
    #    f.write(vstr + '\n')
    #    j += 1
    #f.close()

    


    while (l<config.lmax):
        while(n<config.nmax):

            f=open("z.txt", "w")
#           if n==1:
#                E_l= guess
#                Z_l=Shootsolve(v, xgrid, l, -1.0/guess)
#            else:
#                E_l= eigvals[0, l, n-1] + jump 
#                Z_l=Shootsolve(v, xgrid, l, -1.0/E_l)
            print('n=',n)
            if(n<=l): #not physical
                break
            else:
                if(n==1): #start searching at the guess value #QZQZQZn==1
                    E_0=guess
                    Z_0=Shootsolve(v, xgrid, l, -1.0/E_0)
                elif(n==2): #After one eigenvalue is found, start searching from it #QZQZQZ n==2
                    E_0=-1.0/eigvals[0, l, 1]+st
                    Z_0=Shootsolve(v, xgrid, l, -1.0/E_0)
                elif(n>2):#after at least 2 eigenvalue are found. The search can start from the last eigenvalue+ jump. The jump is based on the observation that - #QZQZQZ n>2
                    E_0=-1.0/eigvals[0, l, n-1]+jump # - the difference between two energy levels is always smaller than previous 
                    Z_0=Shootsolve(v, xgrid, l, -1.0/E_0)
    
                count=1
                while (count<max_count): #Searching for energy values:
                    E_1=E_0+st
                    Z_1=Shootsolve(v, xgrid, l, -1.0/E_1)
                    #print('E_1',-1.0/E_1, 'Z_1', Z_1 )
                    zstring=str(Z_1)
                    f.write(zstring + '\n')
                    if(Z_1*Z_0<=0.0): #If a root of the function Z is bracketed, use the refine function to calculate it
                        root, E  = refine(v, xgrid, l, E_0, E_1)
                        if root==True:
                            eigvals[0, l, n] = E
                            #print('E=', E)
                            break
                       # elif root==False:
                       #     print('False Zero')
                       #     continue
                    count = count+1
                    E_0=E_1
                    Z_0=Z_1
                

                Psi=Shootwrite(v, xgrid, l, eigvals[0, l, n]) #Wrtie the wavefunction
                i=0
               #print('QQQQ', eigvals[0, l, n])
                while i<int(np.size(xgrid)): #Copy WF to output array
                    eigfuncs[0, l, n, i] = Psi[i]
                   # eigfuncs[0, l, n, i] = D[i]
                    i=i+1
                i=0
                while i<int(np.size(xgrid)):
                    if Psi[i] != 0:
                        #print('!')
                        break
                    i+=1

                if (n>1):
                    jump=0.8*(-1.0/eigvals[0, l, n]+1.0/eigvals[0, l, n-1]) #Calculates the jump as 0.8 times the (inverse) of the energy difference
                
            n=n+1
        l=l+1
        jump=0.0
        eigfuncs=np.sort(eigfuncs, axis=2)
        f.close()
    return eigfuncs, eigvals

def refine(v, xgrid, l,  x_0, x_1):
    #When an eigenvalue is bracketed, this subroutine will refine the bracketing until a root is found or the bracket is smaller then a user defined energy error.
    #The refinement process is done via the secent method.
    flag1=False #Internal flag for an in-function loop. When it is true the main loop breaks
    flag2=False #Reutrns true if a zero is found. Returns False if the bracket does not contain a zero.
    dx=xgrid[1]-xgrid[0]
    h4=dx**4
    E_err=10**(-4) #Maximal energy error
    
    

    while(flag1==False):
        y_1= Shootsolve(v, xgrid, l, -1.0/x_1)
        y_0= Shootsolve(v, xgrid, l, -1.0/x_0)
        
        #x_2= x_1-y_1*(x_1-x_0)/(y_1-y_0) #The mid-point defined by the secent method
        x_2=(x_0+x_1)/2.0
        y_2= Shootsolve(v, xgrid, l, -1.0/x_2)
        #print('find',-1.0/x_0, -1.0/x_1)   

        #searches if the zero is in [x_0,x_2] or [x_2,x_1] and then redefines the bracket based on it
        if (y_1*y_2 <=0.0): 
            if abs(x_2-x_0)<E_err:  #If the bracket is to small then then exit function
                if (abs(y_1)<ma.sqrt(dx) and abs(y_0)<ma.sqrt(dx)): 
                    E=-2.0/(x_1+x_2)
                    flag2=True
                    flag1=True
                else:
                    E=0.0
                    flag2=False
                    flag1=True

            y_0=y_2; x_0=x_2
        elif (y_0*y_2 <=0.0):
            if abs(x_2-x_1)<E_err:
                if (abs(y_1)<ma.sqrt(dx) and abs(y_0)<ma.sqrt(dx)):
                    E=-2.0/(x_2+x_0)
                    flag2=True
                    flag1=True
                else:
                    E=0.0
                    flag2=False
                    flag1=True
            
            y_1=y_2; x_1=x_2
        
        #The 'if' conditions for which a zero is declared

        if (abs(y_1-y_0)<h4): 
            E=-2.0/(x_1+x_0)
            flag2=True
            flag1=True

        elif (abs(y_0)<h4):
            E=-1.0/x_0
            flag2=True
            flag1=True
        
        elif (abs(y_1)<h4):
            E=-1.0/x_1
            flag2=True
            flag1=True

        elif (abs(x_1-x_0) < E_err):
            if (abs(y_1)<ma.sqrt(dx) and abs(y_0)<ma.sqrt(dx)):
                E=-2.0/(x_1+x_0)
                flag2=True
                flag1=True
            else:
                E=0.0
                flag2=False
                flag1=True
            
    #print('find', E)
    return flag2, E
    

def Shootsolve(v, xgrid, l, E): #Solves the KS equation by the shooting method for the P_nl. It dosn't write down anything but the value of the derivative continuation.
    #Defining spatial resolution -dx, N-No. of points, k-"Numerov's" k(x)- defined as a zeros array, W- k in the log scale
    #u and y are for performing the integration
    #Note: The array Q[i] is purely to write the function as a check. It has no role in solving the equation
    N = int(np.size(xgrid))
    dx = xgrid[1] - xgrid[0] 
    h=(dx**2)/12.0
    v=v.reshape(-1)
    k=v-E #ZZZ
    W=np.zeros(N, dtype=np.float64)
    Q=np.zeros(N)
    u=np.zeros(3, dtype=np.float64)
    y=np.zeros(3, dtype=np.float64)
    tp=0    
     
    #(i) Setting turning point and calculating the coefficient of P in the DE that we solve (It is NOT W from the preprint):
    W=-2.0*np.exp(2.0*xgrid)*(v-E)-(l+0.5)**2  
    #tp=int(ma.floor(0.85*N))
    i=0
    while i<N:
        if v[i] > (E+v[N-1]):
            tp=i-1
            break
        i += 1
    if tp < 4:
        tp=4
    if tp > N-4:
        tp=N-4
    
    #print(tp, N)
    
    #(ii) Forward integration        
    a=config.grid_params["x0"]  #a is the leftmost grid point. 
    
    u[0]=np.exp((l+0.5)*a)
    u[1]=np.exp((l+0.5)*(a+dx))
    
    i=int(1)
    
    while (i<tp): #Integrating forward using Numerov, until P(x_tp)
        
        u[2]=(2.0*(1.0-5.0*h*W[i])*u[1]-(1.0+h*W[i-1])*u[0])/(1.0+h*W[i+1])
        Q[i]=str(u[2])
        

        u[0]=u[1]
        u[1]=u[2]
        i=i+1
    #Calculating P(x_tp+1)

    y[0]=u[0] 
    y[1]=u[1]
    

    y[2]=(2.0*(1.0-5.0*h*W[int(tp)])*y[1]-(1.0+h*W[int(tp-1)])*y[0])/(1.0+h*W[tp+1])
        
    lef=-(y[2]-y[0])/y[1]
    #print('lef', y[0], y[1], y[2])   
    
    #(iii) Backwards integration
    u[0]=u[1]=u[2]=0.0
    
    #Setting initial conditions

    if (config.bc=="neumann"):#ZZZ Need to check!!
        u[2]=np.exp(-ma.sqrt(-2.0*E)*np.exp(xgrid[N-1])+0.5*xgrid[N-1])
        u[1]=(1-0.5*dx)*u[2]
    elif config.bc=="dirichlet":
        u[2]=0
        u[1]=np.exp(-ma.sqrt(-2.0*E)*np.exp(xgrid[N-1])+0.5*xgrid[N-1])
    
    
    #Integrating backwards using Numerov
    i=int(N-2)
    while (i>tp):
        u[0]=(-(1.0+h*W[i+1])*u[2]+2.0*(1.0-5.0*h*W[i])*u[1])/(1.0+h*W[i-1])
        Q[i]=str(u[2])
        u[2]=u[1]
        u[1]=u[0]
        i=i-1
        
    y[2]=u[1]
    y[1]=u[0]

    y[0]=(-(1.0+h*W[tp+1])*y[2]+2.0*(1.0-5.0*h*W[tp])*y[1])/(1.0+h*W[tp-1])
        
    #print('right', y[0], y[1], y[2])   
    right=(y[0]-y[2])/y[1]
    
    #f= open("test.txt", "w") 
    #i=0
    #while i<N:
    #    qi=str(Q[i])
    #    f.write(qi+'\n')
    #    i += 1
    #f.close()

    #Defining the "differentiability" function for the specific (E,l)
    cont=lef-right
    #print(cont, lef, right, E)
    return cont
            
def Shootwrite(v, xgrid, l, E): #Solves the KS equation for P_nl, normalizes it, writes it down and reports it. 
   #QQQQQ 
    v=v.reshape(-1)
    dx=xgrid[1]-xgrid[0]
    N = int(np.size(xgrid))
    h=(dx**2)/12.0
    k=v-E #ZZZ
    W=np.zeros(N, dtype=np.float64)
    P=np.zeros(N, dtype=np.float64)
    #print("write!", E)

    #(i) Searching for turning point:
    
    W=-2.0*np.exp(2.0*xgrid)*(v-E)-(l+0.5)**2  
    #tp=int(ma.floor(5*N/6))
    i=0
    while i<N:
        if v[i] > (E+v[N-1]):
            tp=i-1
            break
        i += 1
    
    #allocating 2 arrays - one for (ii) and one for (iii):
    P_lef=np.zeros(tp+1, dtype=np.float64)
    P_rig=np.zeros(N-tp+1, dtype=np.float64)


    #(ii) Forward integration        
    a=config.grid_params["x0"]  #a is the leftmost grid point. 
    P_lef[0]=ma.exp((l+0.5)*a)
    P_lef[1]=ma.exp((l+0.5)*a)*(1+dx*(l+0.5))
    i=int(2)
    while (i<=tp):

        P_lef[i]=(2.0*(1.0-5.0*h*W[i-1])*P_lef[i-1]-(1.0+h*W[i-2])*P_lef[i-2])/(1.0+h*W[i])
        i=i+1
        

    #P[tp]=(2.0*(1.0-5.0*h/12.0*W[tp-1])*P[tp-1]-(1.0+h/12.0*W[tp-2])*P[tp-2])/(1.0+h/12.0*W[tp])
    #check=P[tp]

    #(iii) Backwards integration
        
        
    if (config.bc=="neumann"):#ZZZ Need to check!!
        P_rig[N-tp]=np.exp(-ma.sqrt(-2.0*E)*np.exp(xgrid[N-1])+0.5*xgrid[N-1])
        P_rig[N-tp-1]=(1-0.5*dx)*P_rig[N-tp]
    elif config.bc=="dirichlet":
        P_rig[-1]=0.0
        P_rig[-2]=np.exp(-ma.sqrt(-2.0*E)*np.exp(xgrid[N-1])+0.5*xgrid[N-1])
    #print('1', P_rig[-1]) 
    i=int(N-tp-2)
    while (i>=0):
        P_rig[i]=(-(1.0+h*W[i+2])*P_rig[i+2]+2.0*(1.0-5.0*h*W[i+1])*P_rig[i+1])/(1.0+h*W[i])
        i=i-1
    
    #imposing continuity
    h_l=P_lef[tp]
    h_r=P_rig[0]
    P_rig=(h_l/h_r)*P_rig
    #print('2',P_rig[-1])

    i=int(0)
    while (i<N):
        if (i<=tp):
            P[i]=P_lef[i]
        elif (i>tp):
            P[i]=P_rig[i-tp+1]
        i=i+1

    P = np.concatenate((P_lef,P_rig[2:]))
    #print(np.shape(P_lef), np.shape(P_rig))
    
    #print('3', P[-1])
    #P_norm=mathtools.normalize_orbs(P,xgrid)
    P_norm = P
    return P_norm


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
