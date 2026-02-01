import setuptools

setuptools.setup(
    name="crosspy", 
    version="0.0.1b",
    author="Vladislav Myrov",
    author_email="vladislav.myrov@aalto.fi",
    description="Cross-analytics package",
    # long_description=long_description,
    # long_description_content_type="text/markdown",
    url="https://version.helsinki.fi/vlamyr/cross_analytics",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "statsmodels",
        "numba",
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
        "nibabel",
        "Pillow",
        "datashader",
        "pyvista",
        "mne",
        "joblib",
        "networkx",
        "powerlaw"
    ]
)
