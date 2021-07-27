import setuptools
    
setuptools.setup(
    name="CWT", 
    version=2,
    description="A tensorflow 2.0 Continuous Wavelet Transform",
    long_description=open('README.md').read(),
    packages=['CWT'],
    install_requires=['numpy', 'tensorflow'],
    python_requires='>=3.6',
)
