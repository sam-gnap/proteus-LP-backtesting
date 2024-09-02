from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="proteus-LP-backtesting",
    version="0.1.0",
    author="Your Name",
    author_email="",
    description="A backtesting tool for Liquidity Provider strategies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sam-gnap/proteus-LP-backtesting",
    packages=find_packages(include=['src', 'src.*', 'config']),
    package_data={'config': ['*.py']},
    install_requires=[
        "requests",
        "matplotlib",
        "python-dotenv",
        # Add other dependencies here
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.7",
)
