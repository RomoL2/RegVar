from setuptools import setup
from setuptools.extension import Extension
import RBPamp

try:
    from Cython.Distutils import build_ext
    from Cython.Build import cythonize
except ImportError:
    use_cython = False
else:
    use_cython = True

cmdclass = { }
ext_modules = [ ]


if use_cython:
    import numpy
    import os

    if os.uname().sysname == 'Darwin':
        # Mac OS X uses LLVM/Clang and openmp works slightly differently compared to gcc
        cy_kw = dict(
            include_dirs=[numpy.get_include(), ],
            extra_compile_args=['-O3', '-ffast-math', '-march=native', '-mtune=native'], 
            extra_link_args=['-lomp'],
            language_level="3"
        )
    else:
        # All the linuxes AFAIK use gcc by default. Perhaps check should be on which compiler is used?
        cy_kw = dict(
            include_dirs=[numpy.get_include(), ],
            extra_compile_args=['-fopenmp', '-O3', '-ffast-math', '-march=native', '-mtune=native'], 
            extra_link_args=['-fopenmp'],
            language_level="3"
        )

    ext_modules += [
        Extension("RBPamp.cy.cy_kmers", [ "RBPamp/cython/kmers.pyx" ], **cy_kw),
        Extension("RBPamp.cy.cy_model", [ "RBPamp/cython/model.pyx" ], **cy_kw ),
        Extension("RBPamp.cy.cy_fastrand", [ "RBPamp/cython/fastrand.pyx" ], **cy_kw ),
        #Extension("RBPamp.cy_cmpxchg", [ "RBPamp/cython/test_cmpxchg.pyx" ], extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp'], ),
    ]

    cmdclass.update({ 'build_ext': build_ext })
else:
    ext_modules += [
        Extension("RBPamp.cy.kmers", [ "RBPamp/cython/kmers.c" ]),
        Extension("RBPamp.cy.model", [ "RBPamp/cython/model.c" ]),
        Extension("RBPamp.cy.fastrand", [ "RBPamp/cython/fastrand.c" ]),
    ]

desc = """ 
RNA-Binding Protein Affinity Model with Physical constraints

A biophysical model and fit for RNA bind'n'seq (RBNS) experiments 
(Lambert et al. 2014, Dominguez et al. 2018). Yields a compact, 
versatile, and predictive description of an RNA-binding proteins 
primary sequence affinity landscape.

Publications: Jens & Burge 2020 (in preparation)',
"""

setup(
    name = RBPamp.__name__,
    version = RBPamp.__version__,
    description=desc,
    url = 'https://bitbucket.org/marjens/RBPamp/',
    author = 'Marvin Jens',
    author_email = 'mjens@mit.edu',
    license = 'MIT',
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
        # 'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
    keywords = 'rna RBNS k-mer kmer statistics biology bioinformatics RBP RNA-binding protein gene regulation affinity thermodynamics gradient descent',

    install_requires=['cython', 'numpy', 'pandas', 'matplotlib', 'seaborn', 'zmq', 'jinja2', 'future_fstrings'],
    scripts=['bin/RBPamp'],
    package_dir='',
    packages=['RBPamp'],
    cmdclass = cmdclass,
    ext_modules=ext_modules,
)
