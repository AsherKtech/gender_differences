from setuptools import setup, Extension
import pybind11

setup(
    ext_modules=[
        Extension(
            'group_valid',
            ['group_valid.cpp'],
            include_dirs=[pybind11.get_include()],
            language='c++',
            extra_compile_args=['-std=c++11']
        )
    ]
)
