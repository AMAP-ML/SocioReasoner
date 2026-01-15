import os
import re

from setuptools import find_packages, setup


def get_version():
    with open(os.path.join("src", "mcore_adapter", "__init__.py"), "r", encoding="utf-8") as f:
        file_content = f.read()
        pattern = r"{0}\W*=\W*\"([^\"]+)\"".format("__version__")
        (version,) = re.findall(pattern, file_content)
        return version


def get_requires():
    with open("requirements.txt", "r", encoding="utf-8") as f:
        file_content = f.read()
        lines = [line.strip() for line in file_content.strip().split("\n") if not line.startswith("#")]
        return lines


setup(
    name="mcore_adapter",
    version=get_version(),
    description="",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=get_requires(),
    python_requires=">=3.8.13",
)
