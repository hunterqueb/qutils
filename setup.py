# Created 12/9/23
# Author: Hunter Quebedeaux
from skbuild import setup
from setuptools import find_packages

setup(
      name="quebutils",
      version="0.1",
      description="General utilities for engineering research in Python.",
      author='Hunter Quebedeaux',
      packages=find_packages(),
      python_requires=">=3.8",
)