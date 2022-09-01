import os

from setuptools import setup


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="tf_msn",
    version="0.0.1",
    author="Aritra and Ayush",
    author_email="mein2work@gmail.com",
    description=(
        "Implementation of Masked Siamese Networks for Label-Efficient Learning (https://arxiv.org/abs/2204.07141) in TensorFlow."
    ),
    license="MIT License",
    keywords="ssl tensorflow keras transformer",
    packages=["msn", "tests", "configs"],
    long_description=read("README.md"),
)
