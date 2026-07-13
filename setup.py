from setuptools import setup
from pathlib import Path

setup(
    name="FRAP",
    version="1.0.0.post02",
    author="Tomohiro C. Yoshida",
    author_email="tomohiroyoshida.astro@gmail.com",
    description="Flexible Radial Analysis of Protoplanetary disks. A Python module for astronomical data analysis.",
    packages=["frap"],
    project_urls={
        "Documentation": "https://frap.readthedocs.io/en/latest/index.html",
        "Source Code": "https://github.com/tomyoshida/frap"
    },
)
