#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
import numpy as np

import modules

def main():
	path = '/home/victor/Documents/msc-image-database/callisia/alligned/stack/'
	# path = '/home/victor/Documents/repositories/github/python/light-microscopy-image-fusion-prototype/images/beer-stack/'
	# path = '/home/victor/Documents/repositories/github/python/light-microscopy-image-fusion-prototype/images/lena/'

	fusion = modules.Fusion()
	fusion.run(path)


if __name__ == "__main__":
	main()