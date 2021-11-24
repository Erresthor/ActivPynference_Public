import setuptools

setuptools.setup(
    name="activ_pynference",
    version="0.0.1",
    author="activ_pynference",
    author_email="annicchiarico.come@gmail.com",
    description= ("A Python package for solving Markov Decision Processes with Active Inference"),
    license='MIT',
    url="",
    python_requires='>3.7',
    install_requires =[],
    packages=[
        "apynf",
        "apynf.base",
        "apynf.deep"
    ],
    include_package_data=True,
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
)