# http://peterdowns.com/posts/first-time-with-pypi.html
# git tag x.y.z
# git push --tags
# python setup.py sdist upload -r pypi

import os
from distutils.core import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
  name = 'inventoryanalytics',
  packages = ['inventoryanalytics',
              'inventoryanalytics.forecasting',
              'inventoryanalytics.forecasting.test',
              'inventoryanalytics.lotsizing',
              'inventoryanalytics.lotsizing.deterministic',
              'inventoryanalytics.lotsizing.deterministic.constant',
              'inventoryanalytics.lotsizing.deterministic.constant.test',
              'inventoryanalytics.lotsizing.deterministic.time_varying',
              'inventoryanalytics.lotsizing.deterministic.time_varying.test',
              'inventoryanalytics.lotsizing.stochastic',
              'inventoryanalytics.lotsizing.stochastic.nonstationary',
              'inventoryanalytics.lotsizing.stochastic.nonstationary.ss_policy',
              'inventoryanalytics.lotsizing.stochastic.nonstationary.test',
              'inventoryanalytics.lotsizing.stochastic.stationary',
              'inventoryanalytics.lotsizing.stochastic.stationary.test',
              'inventoryanalytics.simulation',
              'inventoryanalytics.simulation.deterministic',
              'inventoryanalytics.simulation.stochastic',
              'inventoryanalytics.utils',
              'inventoryanalytics.utils.test'], # this must be the same as the name above
  version = '1.1',
  description = 'A Python library dedicated to Inventory Analytics.',
  long_description= 'A Python library dedicated to Inventory Analytics.', #read('README.md'), This cannot be used since distutils does not automatically include README.md in MANIFEST
  author = 'Roberto Rossi',
  author_email = 'robros@gmail.com',
  url = 'https://github.com/gwr3n/inventoryanalytics', # use the URL to the github repo
  download_url = 'https://github.com/gwr3n/inventoryanalytics/archive/1.1.tar.gz', # I'll explain this in a second
  keywords = ['inventory', 'analytics'], # arbitrary keywords
  classifiers = [],
)