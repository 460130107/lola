from setuptools import setup
import sys

try:
    import numpy as np
except ImportError:
    print('First you need to install numpy!', file=sys.stderr)
    sys.exit(1)

# In the future if we have Cython code we can use this
#try:
#    from Cython.Build import cythonize
#except ImportError:
#    print('First you need to install cython!', file=sys.stderr)
#    sys.exit(1)

#ext_modules = cythonize('**/*.pyx',
#                        language='c++',
#                        exclude=[],
#                        )

setup(
    name='lola',
    license='Apache 2.0',
    authors='Guido Linders and Wilker Aziz',
    description='Log-linear alignment models',
    version='0.0.dev1',
    packages=['lola'],
    install_requires=[],
    include_dirs=[np.get_include()],
#    ext_modules=ext_modules,
)
