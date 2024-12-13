"""Setup script for synapticrag package"""
from setuptools import setup, find_packages

# Read requirements
with open('requirements.txt') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Read test requirements
test_requirements = [
    "pytest>=7.3.0",
    "pytest-cov>=4.0.0"
]

setup(
    name="synapticrag",
    version="0.1.0",
    description="A hybrid RAG system combining LightRAG and MemoRAG approaches",
    author="SynapticRAG Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "test": test_requirements,
    },
    include_package_data=True,
    package_data={
        "synaptic": ["py.typed"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
