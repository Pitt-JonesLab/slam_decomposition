from setuptools import setup, find_packages
import io

name = 'slam'
description = 'Parallel Driving for Fast Quantum Computing Under Speed Limits'

# README file as long_description.
long_description = io.open("README.md", encoding="utf-8").read()

# Read in requirements
requirements = open("requirements.txt").readlines()
requirements = [r.strip() for r in requirements]

packages = find_packages()
print(packages)

setup(
    name=name,
    version="0.0.1",
    author="Evan McKinney",
    author_email="evm33@pitt.edu",
    python_requires=(">=3.7.0"),
    install_requires=requirements,
    description=description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=packages
)