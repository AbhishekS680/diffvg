# Adapted from https://github.com/pybind/cmake_example/blob/master/setup.py
import os
import re
import sys
import platform
import subprocess
import importlib
from sysconfig import get_paths
import importlib
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install
from distutils.sysconfig import get_config_var
from distutils.version import LooseVersion

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir, build_with_cuda):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)
        self.build_with_cuda = build_with_cuda

class Build(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))
        super().run()

    def build_extension(self, ext):
        if isinstance(ext, CMakeExtension):
            extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
            info = get_paths()
            include_path = info['include']
            cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                          '-DPYTHON_INCLUDE_PATH=' + include_path,
                          '-DPYTHON_EXECUTABLE=' + sys.executable,
                          '-DPython_EXECUTABLE=' + sys.executable,
                          '-DPython_ROOT_DIR=' + sys.prefix]
            # get_config_var('LIBDIR') returns None on Windows (it's a Unix
            # concept) -- only pass -DPYTHON_LIBRARY when we actually have a
            # value, otherwise str + None crashes.
            python_libdir = get_config_var('LIBDIR')
            if python_libdir is not None:
                cmake_args.append('-DPYTHON_LIBRARY=' + python_libdir)
            # On Windows, CMake's FindPython can pick up a *different* Python
            # install (e.g. a system-wide one) than the conda env actually
            # running this script, even with Python_ROOT_DIR set. That causes
            # a link failure: the compiled objects reference pythonXY.lib
            # matching THIS interpreter's version, but the linker gets pointed
            # at a different install's libs folder that doesn't have that
            # file. Force it explicitly using this interpreter's own version
            # and prefix, which is unambiguous.
            if platform.system() == "Windows":
                py_lib_name = 'python{}{}.lib'.format(sys.version_info.major, sys.version_info.minor)
                py_lib_path = os.path.join(sys.prefix, 'libs', py_lib_name)
                if os.path.exists(py_lib_path):
                    cmake_args.append('-DPython_LIBRARY=' + py_lib_path)
                    cmake_args.append('-DPYTHON_LIBRARY=' + py_lib_path)
                else:
                    print('WARNING: expected Python import library not found at {} -- '
                          'link step may fail or pick the wrong Python.'.format(py_lib_path))
            cfg = 'Debug' if self.debug else 'Release'
            build_args = ['--config', cfg]

            if platform.system() == "Windows":
                cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir),
                               '-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
                if sys.maxsize > 2**32:
                    cmake_args += ['-A', 'x64']
                build_args += ['--', '/m']
            else:
                cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
                build_args += ['--', '-j8']

            if ext.build_with_cuda:
                cmake_args += ['-DDIFFVG_CUDA=1']
            else:
                cmake_args += ['-DDIFFVG_CUDA=0']

            env = os.environ.copy()
            cxxflags = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                             self.distribution.get_version())
            if platform.system() == "Darwin":
                # Recent AppleClang/Xcode (libc++ from Clang 19+) fully removed
                # the internal _VSTD macro that older thrust releases (up to at
                # least 1.17.x) still rely on in
                # thrust/type_traits/is_contiguous_iterator.h, causing
                # "use of undeclared identifier '_VSTD'". _VSTD used to be a
                # plain alias for std in libc++, so defining it ourselves on
                # the command line reproduces the same behavior without
                # needing to patch the thrust submodule's source directly.
                cxxflags += ' -D_VSTD=std'
            env['CXXFLAGS'] = cxxflags
            if not os.path.exists(self.build_temp):
                os.makedirs(self.build_temp)
            subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
            subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)
        else:
            super().build_extension(ext)

torch_spec = importlib.util.find_spec("torch")
tf_spec = importlib.util.find_spec("tensorflow")
packages = []
build_with_cuda = False
if torch_spec is not None:
    packages.append('pydiffvg')
    import torch
    if torch.cuda.is_available():
        build_with_cuda = True
if tf_spec is not None and sys.platform != 'win32':
    packages.append('pydiffvg_tensorflow')
    if not build_with_cuda:
        import tensorflow as tf
        if tf.test.is_gpu_available(cuda_only=True, min_cuda_compute_capability=None):
            build_with_cuda = True
if len(packages) == 0:
    print('Error: PyTorch or Tensorflow must be installed. For Windows platform only PyTorch is supported.')
    exit()
# Override build_with_cuda with environment variable
if 'DIFFVG_CUDA' in os.environ:
    build_with_cuda = os.environ['DIFFVG_CUDA'] == '1'

setup(name = 'diffvg',
      version = '0.0.1',
      install_requires = ["svgpathtools"],
      description = 'Differentiable Vector Graphics',
      ext_modules = [CMakeExtension('diffvg', '', build_with_cuda)],
      cmdclass = dict(build_ext=Build, install=install),
      packages = packages,
      zip_safe = False)