from setuptools import setup, find_packages

# From requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="fakereviewdetector",
    version="0.1.0",
    packages=find_packages(),
    install_requires=requirements,
    author="Zina",
    description="Библиотека для классификации фиктивных отзывов с использованием трансформеров, BiLSTM и pruning.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/zina-frid/fake_review_detection.git",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    include_package_data=True,
)