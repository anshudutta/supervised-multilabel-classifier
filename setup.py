import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="smc",
    version="0.0.1",
    author="Anshu Dutta",
    author_email="anshu.dutta@gmail.com",
    description="Supervised multi-class multi-label classification using NLP",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/anshudutta/supervised-multilabel-classifier/tree/master/supervised_multilabel_classifier",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)