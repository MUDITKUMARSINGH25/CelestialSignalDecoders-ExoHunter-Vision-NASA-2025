from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip()]

setup(
    name="exohunter-vision",
    version="1.0.0",
    author="Celestial Signal Decoders",
    author_email="team@exohuntervision.com",
    description="Multi-Modal AI for Exoplanet Detection - NASA Space Apps 2025",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ExoHunter-Vision-NASA-2025",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    keywords="exoplanet, astronomy, ai, machine-learning, nasa",
)
