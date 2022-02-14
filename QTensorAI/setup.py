from setuptools import setup, find_packages

setup(name='qtensor-ai',
      version='0.1',
      author='Henry Liu',
      author_email='mliu6@uchicago.edu',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'numpy',
          'tqdm'
      ])
