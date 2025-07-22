from setuptools import setup, find_packages

setup(
    name='mif',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "mif": ["data/*.mtx"],
    },
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'networkx>=2.6',
        "importlib-resources; python_version<'3.9'"
        # These libraries are currently all you need.
    ],
    author='Hiroyuki Akama',
    author_email='akamalab01@gmail.com',
    description='Repository for calculating MiF (Markov inverse F-measure) as a sophisticated measure of similarity (distance) between nodes in complex networks',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/hilolani/mif',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
)
