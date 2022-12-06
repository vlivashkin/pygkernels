from setuptools import setup

setup(
    name="pygkernels",
    version="0.9.1",
    packages=["pygkernels"],
    url="",
    license="MIT",
    author="Vladimir Ivashkin",
    author_email="vladimir.ivashkin@phystech.edu",
    description="",
    install_requires=[
        "joblib>=1.2.0",
        "matplotlib>=3.6.2",
        "networkx>=2.8.8",
        "numpy>=1.23.5",
        "pandas>=1.5.2",
        "scikit-learn>=1.1.3",
        "scipy>=1.9.3",
        "tqdm>=4.64.1",
        "torch>=1.12.0",
        "powerlaw>=1.5",
    ],
)
