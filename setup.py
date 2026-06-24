from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "reversi_env",
        ["bindings.cpp"],
        include_dirs=["."],
        cxx_std=17,
    ),
]

setup(
    name="reversi_env",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
