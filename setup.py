import setuptools

with open("README.rst", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='sparsedlist',
    version='0.2',
    packages=[''],
    url='https://github.com/bdragon300/sparsedlist',
    license='Apache-2.0',
    author='Igor Derkach',
    author_email='gosha753951@gmail.com',
    description='Endless list with non-contiguous indexes',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    install_requires=[
        'pyskiplist'
    ]
)
