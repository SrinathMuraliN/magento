import os
import os.path as op
from distutils.core import setup

from setuptools import PEP420PackageFinder

ROOT = op.dirname(op.abspath(__file__))
SRC = op.join(ROOT, "src")


def get_version_info():
    """Extract version information as a dictionary from version.py."""
    version_info = {}
    version_filename = os.path.join("src", "magento_lib", "version.py")
    with open(version_filename, "r") as version_module:
        version_code = compile(version_module.read(), "version.py", "exec")
    exec(version_code, version_info)
    return version_info


setup(
    name="magento_lib",
    version=get_version_info()["version"],
    package_dir={"": "src"},
    description="Magento ML Workflow",
    author="Magento",
    packages=PEP420PackageFinder.find(where=str(SRC)),
)
