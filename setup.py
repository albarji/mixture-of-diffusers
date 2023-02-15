import os
from setuptools import setup


# Read long description from readme
with open("README.md", "r", encoding="utf-8") as fh:
    LONG_DESCRIPTION = fh.read()


# Get tag from Github environment variables
TAG = os.environ['GITHUB_TAG'] if 'GITHUB_TAG' in os.environ else "0.0.0"


setup(
    name="mixdiff",
    version=TAG,
    description="Mixture of Diffusers for scene composition and high resolution image generation .",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    packages=['mixdiff'],
    install_requires=[
        'numpy>=1.19,<2',
        'torch>=1.9,<2',
        'torchvision>=0.10,<1',
        'tqdm>=4.62,<5',
        'scipy==1.10.*',
        'diffusers[torch]==0.7.*',
        'ftfy==6.1.*',
        'gitpython==3.1.*',
        'ligo-segments==1.4.*',
        'torchvision==0.14.*',
        'transformers==4.21.*'
    ],
    author="Alvaro Barbero",
    url='https://github.com/albarji/mixture-of-diffusers',
    license='MIT',
    classifiers=[
        'Development Status :: 1 - Planning',
        'Environment :: GPU :: NVIDIA CUDA',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: Unix',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Artistic Software',
        'Topic :: Multimedia :: Graphics'
    ],
    keywords='artificial-intelligence, deep-learning, diffusion-models',
    test_suite="pytest",
)