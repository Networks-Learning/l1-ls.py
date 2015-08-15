# -*- coding: utf-8 -*-

# This file is part of l1ls.
# https://github.com/musically-ut/l1-ls.py

# Licensed under the MIT license:
# http://www.opensource.org/licenses/MIT-license
# Copyright (c) 2015, Utkarsh Upadhyay <musically.ut@gmail.com>

import numpy as np
import scipy.sparse as S
from numpy.testing import assert_allclose
from preggy import expect
from l1ls import l1ls, l1ls_nonneg

# Example taken from the Matlab version
A = np.array([[1, 0, 0, 0.5], [0, 1, 0.2, 0.3], [0, 0.1, 1, 0.2]])
x0 = np.array([1, 0, 1, 0], dtype='f8')  # Original signal
y = A.dot(x0)                            # noise free signal
lmbda = 0.01                             # regularization parameter
rel_tol = 0.01

# Answers expected
answer = np.array([0.993010, 0.00039478, 0.994096, 0.00403702])
answer_nonneg = np.array([0.9916731, 0.0024232, 0.9928389, 0.0066933])
answer_high_accuracy = np.array([9.9472e-01, 1.0040e-04,
                                 9.9503e-01, 5.5977e-04])
answer_high_accuracy_nonneg = np.array([9.9420e-01, 3.6563e-04,
                                        9.9469e-01, 1.6079e-03])


def test_small_example():
    [x, status, hist] = l1ls(A, y, lmbda, tar_gap=rel_tol)
    assert_allclose(x, answer, atol=1e-5)
    expect(hist.shape).to_equal((12, 5))


def test_small_example_nonneg():
    [x, status, hist] = l1ls_nonneg(A, y, lmbda, tar_gap=rel_tol)
    assert_allclose(x, answer_nonneg, atol=1e-5)
    expect(hist.shape).to_equal((12, 5))


def test_small_example_sparse():
    [x, status, hist] = l1ls(S.csr_matrix(A), y, lmbda, tar_gap=rel_tol)
    assert_allclose(x, answer, atol=1e-5)
    expect(hist.shape).to_equal((12, 5))


def test_small_example_sparse_nonneg():
    [x, status, hist] = l1ls_nonneg(S.csr_matrix(A), y, lmbda, tar_gap=rel_tol)
    assert_allclose(x, answer_nonneg, atol=1e-5)
    expect(hist.shape).to_equal((12, 5))


def test_high_accuracy():
    [x, status, hist] = l1ls(A, y, lmbda, tar_gap=rel_tol / 10)
    assert_allclose(x, answer_high_accuracy, atol=1e-5)
    expect(hist.shape).to_equal((16, 5))


def test_high_accuracy_nonneg():
    [x, status, hist] = l1ls_nonneg(A, y, lmbda, tar_gap=rel_tol / 10)
    assert_allclose(x, answer_high_accuracy_nonneg, atol=1e-5)
    expect(hist.shape).to_equal((15, 5))


def test_initial_value():
    [x, status, hist] = l1ls(A, y, lmbda, x0=answer, tar_gap=rel_tol)
    assert_allclose(x, answer, atol=1e-5)
    expect(x.shape).to_equal(answer.shape)
    expect(hist.shape[0]).to_equal(1)


def test_initial_value_shape():
    shapedAns = answer.reshape(-1, 1)
    [x, status, hist] = l1ls(A, y, lmbda, x0=shapedAns, tar_gap=rel_tol)
    assert_allclose(x, shapedAns, atol=1e-5)
    expect(x.shape).to_equal(shapedAns.shape)
    expect(hist.shape[0]).to_equal(1)


def test_shaped_y():
    # Shape of 'y' does not make a difference
    [x, status, hist] = l1ls(A, y.reshape(-1, 1), lmbda, tar_gap=rel_tol)
    expect(x.shape).to_equal((A.shape[1],))
