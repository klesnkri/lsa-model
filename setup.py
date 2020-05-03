from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
    name="lsa",
    version="0.1",
    description="Latent Semantic Analysis",
    long_description=long_description,
    author="David Mašek, Kristýna Klesnilová",
    packages=["lsa"],
    install_requires=['numpy', 'pandas', 'nltk', 'tqdm']
)
