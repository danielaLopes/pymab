from setuptools import setup, find_packages

setup(
    name="pymab",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        # Add your dependencies here
        # e.g., 'numpy', 'pandas'
    ],
    include_package_data=True,
    package_data={
        'pymab': ['pymab/logging.json'],  # Include the logging configuration file
    },
    author="Daniela Lopes",
    author_email="danielalopes_97@hotmail.com",
    description="A Python library for reinforcement learning algorithms.",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/danielaLopes/pymab",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)