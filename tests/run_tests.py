import unittest
import os


def run_all_tests():
    # Set the test directory to the current directory ('.') since this script is in the 'tests' folder
    test_dir = '.'

    # Discover all test cases in the current directory
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=test_dir, pattern="test_*.py")

    # Run the test suite with verbosity level 2 for detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)


if __name__ == '__main__':
    run_all_tests()
