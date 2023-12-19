import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


__version__ = "0.0.0"

GITHUB_REPO_NAME = "BBC_News_Classification_LSTM"
GITHUB_AUTHOR_USER_NAME = "SunilKumar-ugra"
SRC_REPO = "bbc_news"
GITHUB_AUTHOR_EMAIL = "ugargolsunilkumar@gmail.com"



setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=GITHUB_AUTHOR_USER_NAME,
    author_email=GITHUB_AUTHOR_EMAIL,
    description="BBC News Classification Using LSTM",
    long_description=long_description,
    long_description_content="text/markdown",
    url=f"https://github.com/{GITHUB_AUTHOR_USER_NAME}/{GITHUB_REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{GITHUB_AUTHOR_USER_NAME}/{GITHUB_REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")
)
