from setuptools import setup, find_packages

setup(
    name="NBRsimulator",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        # List dependencies here, e.g.,
        "numpy",
        "matplotlib",
        "scipy"
    ],
    author="Sadman Ahmed Shanto, James Farmer",
    description="A simulator for quasiparticle trapping dynamics using nanobridge resonators.",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
