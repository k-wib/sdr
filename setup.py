import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sdr",
    version="1.0.0",
    author="...",
    author_email="...",
    description="Gaussian Mixture Models Sufficient Dimension Reduction.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/k-wib/sdr',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['numpy','sklearn','matplotlib','torch==1.0.2']
) 
