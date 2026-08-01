from pathlib import Path

import numpy
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup


ROOT = Path(__file__).resolve().parent

extensions = [
    Extension(
        "ms2spectra.frag.compute_frags",
        [str(ROOT / "code/src/ms2spectra/frag/compute_frags.pyx")],
        include_dirs=[numpy.get_include()],
    ),
]

setup(
    package_dir={"": "code/src"},
    packages=find_packages("code/src"),
    ext_modules=cythonize(extensions, language_level=3),
)
