import setuptools

setuptools.setup(
    name='active_pynference',
    version='0.1.25',    
    description='A Python implementation of an Active Inference engine using Sophisticated Inference.',
    url='https://github.com/Erresthor/ActivPynference_Public',
    author='Come Annicchiarico',
    author_email='come.annicchiarico@inserm.fr',
    python_requires='>3.7',
    license='MIT',
    packages=setuptools.find_packages(),
    install_requires=['numpy',
                    'scipy',
                    'matplotlib',
                    'pillow'                
                    ],
    keywords=[
        "artificial intelligence",
        "active inference",
        "free energy principle"
        "information theory",
        "decision-making",
        "MDP",
        "Markov Decision Process",
        "Bayesian inference",
        "variational inference",
        "reinforcement learning"
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
    ],
    include_package_data=True
)