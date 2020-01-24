#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""This module's docstring summary line.

This is a multi-line docstring. Paragraphs are separated with blank lines.
Lines conform to 79-column limit.

Module and packages names should be short, lower_case_with_underscores.
Notice that this in not PEP8-cheatsheet.py

Seriously, use flake8. Atom.io with https://atom.io/packages/linter-flake8
is awesome!

See http://www.python.org/dev/peps/pep-0008/ for more PEP-8 details
"""

import numpy as np
import skimage


class Evaluation(object):
    """Class with pre-processing operations for fusion rules."""

    def __init__(self):
        super().__init__()

    def open_image(self, path):
        """Method and function names are lower_case_with_underscores.

        Always use self as first arg.
        """
        img = Image.open(path)
        return img