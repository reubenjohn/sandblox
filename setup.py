from setuptools import setup, find_packages

requires = [
	'tensorflow>=1.8,<1.13',
	'typing==3.6.6',
]

setup(
	name='sandblox',
	version='0.1.1',
	author='Reuben John',
	author_email='reubenvjohn@gmail.com',
	description='Declarative programming framework for graph computing libraries',
	long_description=open('README.md').read(),
	long_description_content_type="text/markdown",
	url='https://github.com/SandBlox/sandblox',
	packages=find_packages(exclude=['sandblox.pythonic_tf', 'sandblox.test']),
	setup_requires=['pytest-runner'],
	install_requires=requires,
	tests_require=['pytest>=2.8.0',],
	classifiers=[
		"Programming Language :: Python :: 3",
		"License :: OSI Approved :: MIT License",
		"Operating System :: OS Independent",
	]
)
