#!/usr/bin/env python
import os
from setuptools import setup


def fopen(fname):
    return open(os.path.join(os.path.dirname(__file__), fname))


def read(fname):
    return fopen(fname).read()

setup(
    name='image-analyzer',
    version='0.1',
    description='Image Analyzer',
    long_description=read('README.md'),
    # author='',
    # author_email='',
    license='MIT',
    # url='',
    include_package_data=True,
    install_requires=[
        line.strip() for line in fopen("requirements.txt")
    ],
    setup_requires=[
    ],
    tests_require=[
    ],
    scripts=[
        'bin/image-analyzer',
    ],
    package_dir={'': 'src'},
    # platforms=["any"],
    platforms=["linux"],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: X11 Applications :: GTK',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        # 'Operating System :: Unix',
        'Programming Language :: Python :: 2 :: Only',
        # "Programming Language :: Python :: 2 :: Only",
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
    ],
)
