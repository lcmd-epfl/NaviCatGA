from distutils.core import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="navicatGA",
    version="0.1",
    author="rlaplaza, lcmd-epfl",
    author_email="laplazasolanas@gmail.com",
    description="A flexible Genetic Algorithm solver",
    long_description=long_description,
    url="https://github.com/lcmd-epfl/NaviCatGA",
    classifiers=["Programming Language :: Python :: 3"],
    packages=["navicatGA"],
    package_dir={"navicatGA": "navicatGA"},
)
