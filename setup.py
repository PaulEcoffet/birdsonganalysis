from setuptools import setup, find_packages

extra = {}

setup(name='birdsonganalysis',
      version='0.0.2',
      packages=[],

      install_requires=['numpy', 'scipy', 'libtfr'],

      setup_requires=['setuptools_git >= 0.3', 'pkgconfig'],

      #   include_package_data=True,
      #   exclude_package_data={'': ['README', '.gitignore']},

      zip_safe=True,

      author='Paul Ecoffet',
      author_email='ecoffet.paul@gmail.com',
      description="Python Library for Bird song analysis, ported from Sound \
Analysis Pro 2011 and Sound Analysis Tools",
      url='https://github.com/PaulEcoffet/birdsonganalysis',
      license='MIT',
      **extra
)
