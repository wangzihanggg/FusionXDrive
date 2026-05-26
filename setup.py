from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="fusionxdrive",
    version="1.0.0",
    author="FusionXDrive Team",
    description="Multi-Modal VLM for Driving Scene Understanding and Trajectory Planning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/FusionXDrive",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.37.0",
        "numpy>=1.24.0",
        "Pillow>=10.0.0",
        "python-lzf>=0.2.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "eval": ["bert-score>=0.3.0", "scipy>=1.10.0"],
        "vis": ["matplotlib>=3.7.0", "opencv-python>=4.8.0", "open3d>=0.17.0"],
        "all": [
            "bert-score>=0.3.0",
            "scipy>=1.10.0",
            "matplotlib>=3.7.0",
            "opencv-python>=4.8.0",
            "open3d>=0.17.0",
        ],
    },
)
