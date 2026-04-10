from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension

ext_modules = [
    Pybind11Extension(
        "random_events_lib",
        [
            "export/bindings.cpp",
            "random_events_lib/src/sigma_algebra.cpp",
            "random_events_lib/src/set.cpp",
            "random_events_lib/src/product_algebra.cpp"
         ],
        include_dirs=["random_events_lib/include"],
        extra_compile_args=["-std=c++17", "-fPIC"],
    ),
]

setup(
    name="random_events_lib",
    version="0.0.1",
    ext_modules=ext_modules,
    zip_safe=False,
)
