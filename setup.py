# http://peterdowns.com/posts/first-time-with-pypi.html
# python setup.py sdist upload -r pypi

from distutils.core import setup
setup(
  name = 'inventoryanalytics',
  packages = ['inventoryanalytics',
              'inventoryanalytics.lotsizing',
              'inventoryanalytics.lotsizing.deterministic',
              'inventoryanalytics.lotsizing.deterministic.constant',
              'inventoryanalytics.lotsizing.deterministic.time_varying',
              'inventoryanalytics.lotsizing.deterministic.time_varying.test',
              'inventoryanalytics.lotsizing.stochastic',
              'inventoryanalytics.lotsizing.stochastic.nonstationary',
              'inventoryanalytics.lotsizing.stochastic.nonstationary.ss_policy',
              'inventoryanalytics.lotsizing.stochastic.nonstationary.test',
              'inventoryanalytics.lotsizing.stochastic.stationary',
              'inventoryanalytics.lotsizing.stochastic.stationary.test',
              'inventoryanalytics.utils',
              'inventoryanalytics.utils.test'], # this must be the same as the name above
  version = '0.5',
  description = 'An Inventory Analytics library.',
  author = 'Roberto Rossi',
  author_email = 'robros@gmail.com',
  url = 'https://github.com/gwr3n/inventoryanalytics', # use the URL to the github repo
  download_url = 'https://github.com/gwr3n/inventoryanalytics/archive/0.5.tar.gz', # I'll explain this in a second
  keywords = ['inventory', 'analytics'], # arbitrary keywords
  classifiers = [],
)