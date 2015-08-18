# -*- coding: utf-8 -*-

# This file is part of l1ls.
# https://github.com/musically-ut/l1-ls.py

# Licensed under the MIT license:
# http://www.opensource.org/licenses/MIT-license
# Copyright (c) 2015, Utkarsh Upadhyay <musically.ut@gmail.com>

from __future__ import print_function
from scipy.sparse.linalg import cg, LinearOperator
import numpy as np
from numpy.linalg import norm


def l1ls_nonneg(A, y, lmbda, x0=None, At=None, m=None, n=None, tar_gap=1e-3,
                quiet=False, eta=1e-3, pcgmaxi=5000):
    """
    Solve a l1-Regularized Least Squares problem when the result is known to
    be non-negative.

    l1ls_nonneg solves problems of the following form:

        minimize ||A*x-y||^2 + lambda*sum|x_i|, x >= 0

    where A and y are problem data and x is variable (described below).

    Parameters
    ----------
    A : mxn matrix
        input data. Columns correspond to features. Pass sparse matrixes in CSR
        format for best performance.
    y : m vector
        outcome.
    lmbda : positive float
        regularization parameter.
    x0: ndarray
        initial guess of the solution
    At : nxm matrix, optional
        transpose of A.
    m : int, optional
        number of examples (rows) of A.
    n : int, optional
        number of features (column)s of A.
    tar_gap : float, optional
        relative target duality gap (default: 1e-3).
    quiet : boolean, optional
        suppress printing message when true (default: False).
    eta : float, optional
        parameter for PCG termination (default: 1e-3).
    pcgmaxi : int, optional
        number of maximum PCG iterations (default: 5000).

    Returns
    -------
    x : array_like
        classifier.
    status  : string
        'Solved' or 'Failed'
    history : matrix
        history data. Columns represent (truncated) Newton iterations; rows
        represent the following:
             - 1st row) gap
             - 2nd row) primal objective
             - 3rd row) dual objective
             - 4th row) step size
             - 5th row) pcg status flag (-1 = error, 1 = failed, 0 = success)

    References
    ----------
    * S.-J. Kim, K. Koh, M. Lustig, S. Boyd, and D. Gorinevsky. An
      Interior-Point Method for Large-Scale l1-Regularized Least Squares,
      (2007), IEEE Journal on Selected Topics in Signal Processing,
      1(4):606-617.
    """
    At = A.transpose() if At is None else At
    m = A.shape[0] if m is None else m
    n = A.shape[1] if n is None else n

    # Interior Point Method parameters
    MU = 2             # updating parameter of t
    MAX_NT_ITER = 400  # maximum number of IPM (Newton) iterations

    # Line search parameters
    ALPHA = 0.01       # minimum fraction of decrease in the objective
    BETA = 0.5         # stepsize decrease factor
    MAX_LS_ITER = 100  # maximum backtracking line search iteration

    t0 = min(max(1, 1/lmbda), n / 1e-3)

    x = np.ones(n) if x0 is None else x0.ravel()
    y = y.ravel()
    status, history = 'Failed', []

    t = t0
    reltol = tar_gap

    f = -x

    # Result/History variables
    pobjs, dobjs, sts, pflgs = [], [], [], []
    pobj, dobj, s, pflg = np.inf, -np.inf, np.inf, 0

    ntiter, lsiter = 0, 0
    normg = 0
    dx = np.zeros(n)

    # This can be slow, so instead, we use a cruder preconditioning
    # diagxtx = diag(At.dot(A))
    diagxtx = 2 * np.ones(n)

    if not quiet:
        print('\nSolving a problem of size (m={}, n={})'
              ', with lambda={:5e}'.format(m, n, lmbda))
        print('----------------------------------------'
              '------------------------------')
        print('{:>5s} {:>9s} {:>15s} {:>15s} {:>13s}'
              .format('iter', 'gap', 'primobj', 'dualobj',
                      'step len'))

    for ntiter in range(0, MAX_NT_ITER):
        z = A.dot(x) - y

        # Calculating the duality gap
        nu = 2 * z

        minAnu = np.min(At.dot(nu))
        if minAnu < -lmbda:
            nu = nu * lmbda / (-minAnu)

        pobj = z.dot(z) + lmbda*np.sum(x)
        dobj = max(-0.25 * nu.dot(nu) - nu.dot(y), dobj)
        gap = pobj - dobj

        pobjs.append(pobj)
        dobjs.append(dobj)
        sts.append(s)
        pflgs.append(pflg)

        # Stopping criterion
        if not quiet:
            print('{:4d} {:12.2e} {:15.5e} {:15.5e} {:11.1e}'
                  .format(ntiter, gap, pobj, dobj, s))

        if (gap / np.abs(dobj)) < reltol:
            status = 'Solved'
            history = np.vstack([np.asarray(pobjs) - np.asarray(dobjs),
                                 pobjs, dobjs, sts, pflgs]).transpose()
            if not quiet:
                print('Absolute tolerance reached.')

            break

        # Update t
        if s >= 0.5:
            t = max(min(n * MU / gap, MU * t), t)

        # Calculate Newton step
        d1 = (1.0 / t) / (x ** 2)

        # calculate the gradient
        gradphi = At.dot(2 * z) + lmbda - (1.0 / t) / x

        # calculate vectors to be used in the preconditioner
        prb = diagxtx + d1

        # set pcg tolerange (relative)
        normg = norm(gradphi)
        pcgtol = min(1e-1, eta * gap / min(1, normg))

        p1 = 1.0 / prb
        dx_old = dx
        [dx, info] = cg(AXfunc(A, At, d1, p1),
                        -gradphi, x0=dx, tol=pcgtol, maxiter=pcgmaxi,
                        M=MXfunc(A, At, d1, p1))

        # This is to increase the tolerance of the underlying PCG if
        # it converges to the same solution without offering an increase
        # in the solution of the actual problem
        if info == 0 and np.all(dx_old == dx):
            pcgtol *= 0.1
            pflg = 0
        elif info < 0:
            pflg = -1
            raise TypeError('Incorrectly formulated problem.'
                            'Could not run PCG on it.')
        elif info > 0:
            pflg = 1
            print('Could not converge PCG after {} iterations.'.format(info))
        else:
            pflg = 0

        # Backtracking line search
        phi = z.dot(z) + lmbda * np.sum(x) - np.sum(np.log(-f)) / t
        s = 1.0
        gdx = gradphi.dot(dx)
        for lsiter in range(MAX_LS_ITER):
            newx = x + s * dx
            newf = -newx
            if np.max(newf) < 0:
                newz = A.dot(newx) - y
                newphi = newz.dot(newz) + \
                    lmbda * np.sum(newx) - np.sum(np.log(-newf)) / t
                if newphi - phi <= ALPHA * s * gdx:
                    break
            s = BETA * s

        if lsiter == MAX_LS_ITER - 1:
            print('Could not find optimal point during line search.')
            break

        x, f = newx, newf

    # Reshape x if the original array was a 2D
    if x0 is not None:
        x = x.reshape(*x0.shape)

    return (x, status, history)


def AXfunc(A, At, d1, p1):
    """
    Returns a linear operator which computes A * x for PCG.

        y = hessphi * [x1; x2],

        where hessphi = [ A'*A*2+D1, D2;
                          D2,        D1]
    """

    def matvec(x):
        return At.dot(A.dot(x) * 2) + d1 * x

    N = d1.shape[0]
    return LinearOperator((N, N), matvec=matvec)


def MXfunc(A, At, d1, p1):
    """
    Compute P^{-1}X (PCG)

    y = P^{-1}*x
    """

    def matvec(x):
        return p1 * x

    N = p1.shape[0]
    return LinearOperator((N, N), matvec=matvec)
