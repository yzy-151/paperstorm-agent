import re

from setuptools import setup, find_packages

# Read the content of the README file
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()
    # Remove p tags.
    pattern = re.compile(r"<p.*?>.*?</p>", re.DOTALL)
    long_description = re.sub(pattern, "", long_description)

# Read the content of the requirements.txt file
with open("requirements.txt", encoding="utf-8") as f:
    requirements = f.read().splitlines()


setup(
    name="paperstorm-agent",
    version="5.6.0",
    author="PaperStorm contributors; Stanford OVAL contributors",
    description="PaperStorm Agent: a production-oriented research and RAG agent built on Stanford STORM.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yzy-151/paperstorm-agent",
    license="MIT License",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10,<3.12",
    install_requires=requirements,
    extras_require={
        "benchmarks": ["datasets>=2.18,<3.0"],
    },
)
