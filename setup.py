import setuptools

setuptools.setup(
    name="crocopy", 
    version="0.0.1b",
    author="Vladislav Myrov",
    author_email="vladislav.myrov@aalto.fi",
    description="Cross-analytics package",
    url="https://https://github.com/palvalab/crocopy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
    install_requires=[
        "statsmodels",
        "numba",
        "numpy",
        "scipy",
        "nibabel",
        "Pillow",
        "pyvista",
        "joblib",
        "powerlaw"
    ]
)
