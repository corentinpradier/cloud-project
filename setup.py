from setuptools import find_packages, setup

setup(
    name="cloudproject",
    version="0.1.0",
    description="A classifier for cloud types with a Gradio interface.",
    author="Corentin Pradier",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
