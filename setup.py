#!/usr/bin/env python
import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

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
    ],
    setup_requires=[
    ],
    tests_require=[
    ],
    platforms=["any"],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Operating System :: OS Independent',
    ],
)
