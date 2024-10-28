from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pymab",
    version="0.1.0",
    packages=find_packages(exclude=["tests", "documentation"]),
    install_requires=[
        "joblib",
        "matplotlib",
        "numpy",
        "pandas",
        "plotly",
        "scipy",
        "tqdm",
    ],
    include_package_data=True,
    package_data={
        "pymab": ["pymab/logging.json"],  # Include the logging configuration file
    },
    author="Daniela Lopes",
    author_email="danielalopes_97@hotmail.com",
    description="A Python library for reinforcement learning algorithms.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/danielaLopes/pymab",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
