from setuptools import setup


setup(
    name='chainermin',
    version='1.2.0',
    author='tsurumeso',
    license='MIT License',
    packages=['chainermin',
              'chainermin.functions',
              'chainermin.functions.activation',
              'chainermin.functions.connection',
              'chainermin.functions.evaluation',
              'chainermin.functions.loss',
              'chainermin.functions.math',
              'chainermin.functions.noise',
              'chainermin.initializers',
              'chainermin.links',
              'chainermin.links.connection',
              'chainermin.optimizers',
              'chainermin.utils'],
)
