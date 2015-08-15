# -*- coding: utf-8 -*-

# This file is part of l1ls.
# https://github.com/musically-ut/l1-ls.py

# Licensed under the MIT license:
# http://www.opensource.org/licenses/MIT-license
# Copyright (c) 2015, Utkarsh Upadhyay <musically.ut@gmail.com>

import numpy as np
from numpy.testing import assert_allclose
from l1ls import l1_ls

# Example taken from the Matlab version
A = np.array([[1, 0, 0, 0.5], [0, 1, 0.2, 0.3], [0, 0.1, 1, 0.2]])
x0 = np.array([1, 0, 1, 0], dtype='f8')  # Original signal
y = A.dot(x0)                            # noise free signal
lmbda = 0.01                             # regularization parameter
rel_tol = 0.01

# Answers expected
answer = np.array([0.993010, 0.00039478, 0.994096, 0.00403702])
answer_high_accuracy = np.array([9.9472e-01, 1.0040e-04, 9.9503e-01, 5.5977e-04])


def test_small_example():
    [x, status, hist] = l1_ls(A, y, lmbda, tar_gap=rel_tol)
    assert_allclose(x, answer, atol=1e-5)


def test_high_accuracy():
    [x, status, hist] = l1_ls(A, y, lmbda, tar_gap=rel_tol / 10)
    assert_allclose(x, answer_high_accuracy, atol=1e-5)

