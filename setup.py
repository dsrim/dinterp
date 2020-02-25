
from __future__ import division, absolute_import, print_function

from distutils.core import setup, Extension
import numpy, os

numpy_dir = os.path.join(numpy.get_include(),'numpy')
ext1 = Extension('dinterp._dinterpc',
                 include_dirs=[numpy_dir],
                 sources = ['dinterp/dinterp.c'])

setup(name = 'dinterp',
      packages = ['dinterp'],
      version = '0.9.2',
      description       = "Tools for displacement interpolation",
      author            = "Donsub Rim",
      ext_modules = [ext1]
      )
