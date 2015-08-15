#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of l1ls.
# https://github.com/musically-ut/l1-ls.py

# Licensed under the MIT license:
# http://www.opensource.org/licenses/MIT-license
# Copyright (c) 2015, Utkarsh Upadhyay <musically.ut@gmail.com>

from setuptools import setup, find_packages
from l1ls import __version__

tests_require = [
    'mock',
    'nose',
    'coverage',
    'yanc',
    'preggy',
    'tox',
    'ipdb',
    'coveralls',
    'sphinx',
]

setup(
    name='l1ls',
    version=__version__,
    description='Python package for solving large scale L1 regularized least squares problems.',
    long_description='''
Python package for solving large scale L1 regularized least squares problems.
''',
    keywords='L1 least-squares optimization',
    author='Utkarsh Upadhyay',
    author_email='musically.ut@gmail.com',
    url='https://github.com/someuser/somepackage',
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: Unix',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Operating System :: OS Independent',
    ],
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        # remember to use 'package-name>=x.y.z,<x.y+1.0' notation
        'scipy>=0.16.0,<0.18.0',
        'numpy>=1.9.2,<1.10.0'
    ],
    extras_require={
        'tests': tests_require,
    },
    entry_points={
        'console_scripts': [
        ],
    },
)
