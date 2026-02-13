"""Setup script for datamanagement package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="datamanagement",
    version="0.1.0",
    author="Shihong Chen",
    author_email="schenek@connect.ust.hk",
    description="A flexible data container with query capabilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/CoderPower-Schenek/DataManagement/",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Database",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.9",
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
        "plot": ["matplotlib>=3.0", "seaborn>=0.11"],
        "ml": ["scikit-learn>=1.0"],
    },
    project_urls={
        "Bug Reports": "https://github.com/CoderPower-Schenek/DataManagement/issues",
        "Source": "https://github.com/CoderPower-Schenek/DataManagement",
        "Documentation": "https://github.com/CoderPower-Schenek/DataManagement/blob/main/Document.md",
    },
)
