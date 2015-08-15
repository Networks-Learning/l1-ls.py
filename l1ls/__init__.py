#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of l1ls.
# https://github.com/musically-ut/l1-ls.py

# Licensed under the MIT license:
# http://www.opensource.org/licenses/MIT-license
# Copyright (c) 2015, Utkarsh Upadhyay <musically.ut@gmail.com>

from .l1_ls import l1ls
from l1ls.version import __version__  # NOQA

__all__ = ['__version__', 'l1ls']
