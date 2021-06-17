import setuptools

setuptools.setup(
    name="SealExtraction",
    version="0.0.1",
    packages=setuptools.find_packages(),
    include_package_data=True,
    python_requires='>=3.7',
    install_requires=['sealExtraction'],
    entry_points={
            'console_scripts': [
                'sealExtraction = sealExtraction.sealExtraction:main',
            ],
        },
)
