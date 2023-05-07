import setuptools
import os
import glob
import json

directory = os.path.dirname(os.path.abspath(__file__))

requirementPath = os.path.join(directory, 'requirements.txt')
install_requires = []
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        install_requires = f.read().splitlines()


submodules = []
for dir in glob.glob('src/GPFramework/projects/**/*.json'):
    with open(dir) as f:
        d = json.load(f)
        submodules.append(d["name"])

setuptools.setup(
      name = 'GPFramework',
      version = '1.0',
      author='Jason Zutty',
      author_email='jasonzutty@gmail.com',
      packages = setuptools.find_packages(where='src'),# ['GPFramework'] + ['GPFramework.projects'] + ['GPFramework.projects.' + submodule for submodule in submodules],
      include_package_data=True,
      package_dir = {'': 'src'},
      install_requires = install_requires,
      )
