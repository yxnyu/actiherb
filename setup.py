from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="molecular-active-learning",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Active learning for molecular property prediction using UniMol",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/molecular-active-learning",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
) 