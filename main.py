#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
import numpy as np

import modules

def main():
	path = str(input()).rstrip()
	# path = '/home/victor/Documents/msc-image-database/callisia/alligned/stack/'

	fusion = modules.Fusion()
	fusion.run(path)


if __name__ == "__main__":
	main()