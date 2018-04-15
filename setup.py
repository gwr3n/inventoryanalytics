# http://peterdowns.com/posts/first-time-with-pypi.html
# python setup.py sdist upload -r pypi

from distutils.core import setup
setup(
  name = 'inventoryanalytics',
  packages = ['inventoryanalytics'], # this must be the same as the name above
  version = '0.4',
  description = 'An Inventory Analytics library.',
  author = 'Roberto Rossi',
  author_email = 'robros@gmail.com',
  url = 'https://github.com/gwr3n/inventoryanalytics', # use the URL to the github repo
  download_url = 'https://github.com/gwr3n/inventoryanalytics/archive/0.4.tar.gz', # I'll explain this in a second
  keywords = ['inventory', 'analytics'], # arbitrary keywords
  classifiers = [],
)