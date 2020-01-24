#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
import numpy as np

import modules

def main():
	# path = str(input()).rstrip()
	path = '/home/victor/Documents/repositories/github/python/light-microscopy-image-fusion-prototype/images/'

	fusion = modules.Fusion()
	fusion.run(path)


if __name__ == "__main__":
	main()