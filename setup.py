from setuptools import setup, find_packages


def scm_version():
    def local_scheme(version):
        return version.format_choice("+{node}", "+{node}.dirty")
    return {"version_scheme": "guess-next-dev", "local_scheme": local_scheme}


setup(
    name="nmigen-soc",
    use_scm_version=scm_version(),
    author="whitequark",
    author_email="whitequark@whitequark.org",
    description="System on Chip toolkit for nMigen",
    #long_description="""TODO""",
    license="BSD",
    setup_requires=["setuptools_scm"],
    install_requires=["nmigen", "nmigen-stdio"],
    packages=find_packages(),
    project_urls={
        "Source Code": "https://github.com/m-labs/nmigen-soc",
        "Bug Tracker": "https://github.com/m-labs/nmigen-soc/issues",
    },
)
