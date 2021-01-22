# Copyright 2016-2021 The Van Valen Lab at the California Institute of
# Technology (Caltech), with support from the Paul Allen Family Foundation,
# Google, & National Institutes of Health (NIH) under Grant U24CA224309-01.
# All rights reserved.
#
# Licensed under a modified Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.github.com/vanvalenlab/deepcell-toolbox/LICENSE
#
# The Work provided may be used for non-commercial academic purposes only.
# For any other use of the Work, including commercial use, please contact:
# vanvalenlab@gmail.com
#
# Neither the name of Caltech nor the names of its contributors may be used
# to endorse or promote products derived from this software without specific
# prior written permission.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os

from codecs import open
from distutils.command.build_ext import build_ext as DistUtilsBuildExt

import setuptools
from setuptools import setup, find_packages
from setuptools.extension import Extension


here = os.path.abspath(os.path.dirname(__file__))


about = {}
with open(os.path.join(here, 'deepcell_toolbox', '__version__.py'), 'r', 'utf-8') as f:
    exec(f.read(), about)


with open(os.path.join(here, 'README.md'), 'r', 'utf-8') as f:
    readme = f.read()


class BuildExtension(setuptools.Command):
    description = DistUtilsBuildExt.description
    user_options = DistUtilsBuildExt.user_options
    boolean_options = DistUtilsBuildExt.boolean_options
    help_options = DistUtilsBuildExt.help_options

    def __init__(self, *args, **kwargs):
        from setuptools.command.build_ext import build_ext as SetupToolsBuildExt

        # Bypass __setatrr__ to avoid infinite recursion.
        self.__dict__['_command'] = SetupToolsBuildExt(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._command, name)

    def __setattr__(self, name, value):
        setattr(self._command, name, value)

    def initialize_options(self, *args, **kwargs):
        return self._command.initialize_options(*args, **kwargs)

    def finalize_options(self, *args, **kwargs):
        ret = self._command.finalize_options(*args, **kwargs)
        import numpy
        self.include_dirs.append(numpy.get_include())
        return ret

    def run(self, *args, **kwargs):
        return self._command.run(*args, **kwargs)


extensions = [
    Extension(
        'deepcell_toolbox.compute_overlap',
        ['deepcell_toolbox/compute_overlap.pyx']
    ),
    Extension(
        'deepcell_toolbox.compute_overlap_3D',
        ['deepcell_toolbox/compute_overlap_3D.pyx']
    ),
]

setup(name=about['__title__'],
      version=about['__version__'],
      description=about['__description__'],
      author=about['__author__'],
      author_email=about['__author_email__'],
      url=about['__url__'],
      license=about['__license__'],
      download_url='{}/tarball/{}'.format(
          about['__url__'], about['__version__']),
      cmdclass={'build_ext': BuildExtension},
      install_requires=['cython',
                        'opencv-python-headless<=3.4.9.31',
                        'pandas',
                        'networkx>=2.1',
                        'numpy',
                        'scipy',
                        'scikit-image',
                        'scikit-learn'],
      extras_require={
          'tests': ['pytest<6',
                    'pytest-pep8',
                    'pytest-cov',
                    'pytest-mock']},
      long_description=readme,
      long_description_content_type='text/markdown',
      packages=find_packages(),
      ext_modules=extensions,
      setup_requires=['cython>=0.28', 'numpy>=1.16.4'],
      classifiers=[
          'Programming Language :: Python',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8'])
