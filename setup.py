from distutils.core import setup
from Cython.Build import cythonize
import numpy
# import setuptools
setup(ext_modules = cythonize('genc.pyx'),
      include_dirs=[numpy.get_include()]
      )