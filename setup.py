#nsml: nsml/ml:cuda9.0-cudnn7-tf-1.11torch1.0keras2.2
from distutils.core import setup
import setuptools

print('setup_test.py is running...')

setup(name='PNS_SAMPLE',
      version='1.0',
      install_requires=['opencv-python', 'torch_optimizer==0.0.1a3', 'scikit-learn==0.21.0']
      ) ## install libraries, 'keras==xx.xx'