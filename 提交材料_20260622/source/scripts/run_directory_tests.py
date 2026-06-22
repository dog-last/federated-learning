#!/usr/bin/env python3
"""Discover and run unit tests under a target directory.

Example:
  python scripts/run_directory_tests.py --dir tests
"""

import argparse
import os
import sys
import unittest


def main():
    parser = argparse.ArgumentParser(description="Run unittest discovery for a directory")
    parser.add_argument("--dir", default="tests", help="Directory to discover tests from")
    parser.add_argument("--pattern", default="test*.py", help="Filename pattern for tests")
    parser.add_argument("--verbosity", type=int, default=2, help="unittest verbosity")
    args = parser.parse_args()

    start_dir = os.path.abspath(args.dir)
    if not os.path.isdir(start_dir):
        print(f"Test directory not found: {start_dir}")
        return 2

    suite = unittest.defaultTestLoader.discover(start_dir=start_dir, pattern=args.pattern)
    result = unittest.TextTestRunner(verbosity=args.verbosity).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
