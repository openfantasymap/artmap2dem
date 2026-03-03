"""Setup script for artmap2dem library."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="artmap2dem",
    version="0.1.0",
    author="ArtMap2DEM Team",
    description="Convert artistic georeferenced maps into believable Digital Elevation Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: GIS",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "rasterio>=1.3.0",
        "scipy>=1.7.0",
        "scikit-image>=0.19.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "opencv-python>=4.5.0",
        "Pillow>=8.0.0",
        "tqdm>=4.60.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
        ],
    },
)
