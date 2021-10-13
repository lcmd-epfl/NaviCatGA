from distutils.core import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="navicatGA",
    version="0.1",
    author="R.LAPLAZA",
    author_email="laplazasolanas@gmail.com",
    description="A flexible Genetic Algorithm solver",
    long_description=long_description,
    url="https://github.com/rlaplaza/navicatGA",
    classifiers=["Programming Language :: Python :: 3"],
    packages=["navicatGA"],
    package_dir={"navicatGA": "navicatGA"},
)
