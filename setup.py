from setuptools import setup
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="FRAP",
    version="1.0.0",
    author="Tomohiro C. Yoshida",
    author_email="tomohiroyoshida.astro@gmail.com",
    description="Flexible Radial Analysis of Protoplanetary disks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["frap"],
    project_urls={
        "Documentation": "https://frap.readthedocs.io/en/latest/index.html",
        "Source Code": "https://github.com/tomyoshida/frap"
    },
)
