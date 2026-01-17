from setuptools import setup, find_packages

setup(
    name="naics-github-classifier",
    version="0.1.0",
    description="Automated NAICS classification for GitHub repositories",
    author="Alexander Quispe",
    author_email="your.email@caltech.edu",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "sentence-transformers>=2.2.0",
        "faiss-gpu>=1.7.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "pyarrow>=12.0.0",
        "requests>=2.31.0",
        "tiktoken>=0.5.0",
        "tqdm>=4.65.0",
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0",
        "matplotlib>=3.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "naics-embed=scripts.generate_embeddings:main",
            "naics-index=scripts.build_faiss_index:main",
            "naics-retrieve=scripts.retrieve_by_naics:main",
            "naics-classify=scripts.classify_repos:main",
            "naics-filter=scripts.filter_results:main",
        ],
    },
)
