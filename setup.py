from setuptools import setup

setup(
    name='pygraphs',
    version='0.9',
    packages=['pygraphs'],
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
        'orange3',
        'pandas',
        'scikit-learn',
        'scipy',
        'tqdm',
    ]
)
