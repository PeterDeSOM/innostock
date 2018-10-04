'''Cloud ML Engine package configuration.'''
from setuptools import setup, find_packages

setup(name='innostock',
      version='0.0.1',
      packages=find_packages(),
      description='Innostock keras model on Cloud ML Engine',
      author='Peter Kim',
      author_email='peterkim.som@gmail.com',
      license='MIT',
      install_requires=[
          'keras',
          'h5py'],
      zip_safe=False)