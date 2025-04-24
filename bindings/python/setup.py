from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import os
import sys
import subprocess
import platform

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        # Required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            "-DLLAMA_BUILD_PYBIND=ON",
        ]

        # Configuration
        cfg = "Debug" if self.debug else "Release"
        build_args = ["--config", cfg]

        cmake_args += [f"-DCMAKE_BUILD_TYPE={cfg}"]
        
        # Set platform-specific build arguments
        if platform.system() == "Windows":
            build_args += ["--", "/m"]
        else:
            build_args += ["--", f"-j{os.cpu_count()}"]

        # Build the project
        os.makedirs(self.build_temp, exist_ok=True)
        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=self.build_temp
        )

setup(
    name="llama-cpp-python",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=[CMakeExtension("llama_cpp.llama_cpp", sourcedir="../../")],
    cmdclass={"build_ext": CMakeBuild},
    install_requires=[
        "numpy>=1.20.0",
    ],
    author="llama.cpp Contributors",
    description="Python bindings for llama.cpp",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ggml-org/llama.cpp",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
