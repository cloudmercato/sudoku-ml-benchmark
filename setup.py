from setuptools import setup, find_packages
import sudoku_ml_benchmark


def read_file(name):
    with open(name) as fd:
        return fd.read()

setup(
    name="sudoku-ml-benchmark",
    version=sudoku_ml_benchmark.__version__,
    author=sudoku_ml_benchmark.__author__,
    author_email=sudoku_ml_benchmark.__email__,
    description=sudoku_ml_benchmark.__doc__,
    url=sudoku_ml_benchmark.__url__,
    license=sudoku_ml_benchmark.__license__,
    py_modules=['sudoku_ml_benchmark'],
    packages=find_packages(),
    install_requires=read_file('requirements.txt').splitlines(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        'Operating System :: OS Independent',
        "Programming Language :: Python",
    ],
    long_description=read_file('README.rst'),
    entry_points={'console_scripts': [
        'sudoku-ml-bench = sudoku_ml_benchmark.console:main',
    ]},
)
