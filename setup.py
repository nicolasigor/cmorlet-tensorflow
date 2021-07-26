import setuptools
import codecs
import os.path

with open("README.md", "r") as fh:
    long_description = fh.read()

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()    
    
def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")    
    
setuptools.setup(
    name="CWT", 
    version=2,
    description="A tensorflow 2.0 Continuous Wavelet Transform",
    long_description=open('README.txt').read(),
    packages=['CWT'],
    install_requires=['numpy', 'tensorflow'],
    python_requires='>=3.6',
)
