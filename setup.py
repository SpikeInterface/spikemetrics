from setuptools import setup, find_packages

d = {}
exec(open("spikemetrics/version.py").read(), None, d)
version = d['version']
long_description = open("README.md").read()

pkg_name = "spikemetrics"

setup(
    name=pkg_name,
    version=version,
    author="Cole Hurwitz, Alessio Paolo Buccino, Josh Siegle, Matthias Hennig, Jeremy Magland, Samuel Garcia",
    author_email="cole.hurwitz@gmail.com",
    description="Python toolkit for computing spike sorting metrics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SpikeInterface/spikemetrics",
    packages=find_packages(),
    package_data={},
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'scikit-learn'
    ],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    )
)
