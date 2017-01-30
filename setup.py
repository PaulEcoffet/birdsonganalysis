from setuptools import setup, find_packages

extra = {}

setup(name='birdsonganalysis',
      version='0.0.2',
      packages=[],

      install_requires=['numpy', 'scipy'],

      setup_requires=['setuptools_git >= 0.3', ],

      #   include_package_data=True,
      #   exclude_package_data={'': ['README', '.gitignore']},

      zip_safe=True,

      author='Paul Ecoffet',
      author_email='ecoffet.paul@gmail.com',
      description='Python Library for Bird song analisys',
      url='https://github.com/PaulEcoffet/birdsonganalysis',
      license='MIT',
      **extra
)
