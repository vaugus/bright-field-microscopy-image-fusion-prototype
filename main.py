#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""Entry point to run the image fusion."""
import modules


def main():
    """Receives the path and calls the Facade."""
    path = str(input()).rstrip()
    modules.Fusion().run(path)



if __name__ == "__main__":
    main()
