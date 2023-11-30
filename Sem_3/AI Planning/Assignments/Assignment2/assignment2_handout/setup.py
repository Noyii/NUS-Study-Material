from setuptools import setup, find_packages

setup(name='elevator',
      version='0.0.1',
      install_requires=['gym==0.26.2', 'numpy'],
      packages=find_packages(),
)