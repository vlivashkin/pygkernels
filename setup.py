from setuptools import setup

setup(
    name='pygkernels',
    version='0.9',
    packages=['pygkernels'],
    url='',
    license='MIT',
    author='Vladimir Ivashkin',
    author_email='vladimir.ivashkin@phystech.edu',
    description='',
    install_requires=[
        'joblib',
        'matplotlib',
        'networkx',
        'numpy',
        'pandas',
        'scikit-learn',
        'scipy',
        'tqdm',
        'torch',
        'powerlaw'
    ]
)
