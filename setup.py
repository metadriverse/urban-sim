# Please don't change the order of following packages!
import os
import sys
from os import path

from setuptools import setup, find_namespace_packages  # This should be place at top!

ROOT_DIR = os.path.dirname(__file__)


def get_version():
    context = {}
    with open('./urbansim/version.py', 'r') as file:
        exec(file.read(), context)
    return context['VERSION']


VERSION = get_version()


def is_mac():
    return sys.platform == "darwin"


def is_win():
    return sys.platform == "win32"


assert sys.version_info.major == 3 and sys.version_info.minor >= 6, \
    "python version >= 3.6, <3.12 is required"

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()
packages = find_namespace_packages(
    include=["urbansim"])
print("We will install the following packages: ", packages)

install_requires = [
    "requests",
    "gymnasium>=0.28",
    "numpy>=1.21.6",
    "matplotlib",
    "pandas",
    "pygame",
    "tqdm",
    "yapf",
    "seaborn",
    "tqdm",
    "progressbar",
    "mapbox_earcut",
    "triangle",
    "manifold3d",
    "pillow",
    "pytest",
    "opencv-python",
    "open3d",
    "lxml",
    "scipy",
    "psutil",
    "geopandas",
    "shapely",
    "filelock",
    "Pygments",
]


cuda_requirement = [
    "cuda-python==12.0.0",
    "PyOpenGL==3.1.6",
    "PyOpenGL-accelerate==3.1.6",
    "pyrr==0.10.3",
    "glfw",
]

gym_requirement = [
    "gym>=0.19.0, <=0.26.0"
]

ros_requirement = [
    "zmq"
]

setup(
    name="urbansim",
    python_requires='>=3.6',  # do version check with assert
    version=VERSION,
    packages=packages,
    install_requires=install_requires,
    extras_require={
        "cuda": cuda_requirement,
        "gym": gym_requirement,
        "ros": ros_requirement,
    },
    include_package_data=True,
    license="Apache 2.0",
    long_description=long_description,
    long_description_content_type='text/markdown',
)
