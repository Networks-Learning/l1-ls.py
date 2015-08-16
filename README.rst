l1-ls
=====

|BuildStatus|

This is a large scale L1 regularized Least Square (L!-LS) solver written in
Python. The code is based entirely on the MATLAB code made available on `Stephen Boyd's l1_ls page <http://stanford.edu/~boyd/papers/l1_ls.html>`_.

|L1LSProblem|


Installation
------------

You can install the bleeding edge directly from the source:

::

    pip install git+https://github.com/musically-ut/l1-ls.py.git@master#egg=l1ls


This package is also available on PyPi.

::

    pip install l1ls

Usage
-----

The module exposes two functions:

- ``l1ls(A, y, lmbda, x0=None, At=None, m=None, n=None, tar_gap=1e-3, quiet=False, eta=1e-3, pcgmaxi=5000)``, and,
- ``l1ls_nonneg(A, y, lmbda, x0=None, At=None, m=None, n=None, tar_gap=1e-3, quiet=False, eta=1e-3, pcgmaxi=5000)``

They can be used as follows:

::

    import l1ls as L
    import numpy as np

    A = np.array([[1, 0, 0, 0.5], [0, 1, 0.2, 0.3], [0, 0.1, 1, 0.2]])
    x0 = np.array([1, 0, 1, 0], dtype='f8')  # Original signal
    y = A.dot(x0)                            # noise free signal
    lmbda = 0.01                             # regularization parameter
    rel_tol = 0.01

    [x, status, hist] = L.l1ls(A, y, lmbda, tar_gap=rel_tol)
    # answer_x = np.array([0.993010, 0.00039478, 0.994096, 0.00403702])

If your matrix ``A`` is sparse, pass it in `CSR format <http://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html>`_
format for best performance.

Reference
---------

- S.-J. Kim, K. Koh, M. Lustig, S. Boyd, and D. Gorinevsky. An
  Interior-Point Method for Large-Scale l1-Regularized Least Squares,
  (2007), IEEE Journal on Selected Topics in Signal Processing,
  1(4):606-617.

.. |BuildStatus| image:: https://travis-ci.org/musically-ut/l1-ls.py.svg?branch=master
   :target: https://travis-ci.org/musically-ut/l1-ls.py

.. |L1LSProblem| image:: http://i.imgur.com/YB8JDTX.gif
