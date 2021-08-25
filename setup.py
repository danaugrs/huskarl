from setuptools import setup
from setuptools import find_packages


setup(
	name='huskarl',
	version='0.4',
	description='Deep Reinforcement Learning Framework',
	author='Daniel Salvadori',
	author_email='danaugrs@gmail.com',
	url='https://github.com/danaugrs/huskarl',
	classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
	python_requires='>=3.6',
	install_requires=[
		'cloudpickle',
		'tensorflow==2.5.1',
		'scipy',
	],
	packages=find_packages()
)
