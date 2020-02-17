#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""Entry point to run the image fusion."""
import numpy as np

import modules


def main():
    """Receives the path and calls the Facade."""
    path = str(input()).rstrip()
    fusion = modules.Fusion()
    fusion.run(path)

if __name__ == "__main__":
    main()
