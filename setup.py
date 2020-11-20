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
        'joblib==0.13.2',
        'matplotlib',
        'networkx==2.3',
        'numpy',
        'pandas',
        'scikit-learn',
        'scipy',
        'tqdm',
        'torch',
        'powerlaw'
    ]
)
