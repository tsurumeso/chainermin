from setuptools import setup


setup(
    name='chainermin',
    version='2.0.0',
    author='tsurumeso',
    license='MIT License',
    packages=['chainermin',
              'chainermin.functions',
              'chainermin.functions.array',
              'chainermin.functions.activation',
              'chainermin.functions.connection',
              'chainermin.functions.evaluation',
              'chainermin.functions.loss',
              'chainermin.functions.math',
              'chainermin.functions.noise',
              'chainermin.functions.normalization',
              'chainermin.initializers',
              'chainermin.links',
              'chainermin.links.connection',
              'chainermin.links.normalization',
              'chainermin.optimizers',
              'chainermin.utils'],
)
