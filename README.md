# Deep Reinforcement Learning Nanodegree

### Instroduction

This repository contains the solutions for the following projects:

- [P1 Nativagation](./p1_navigation)


### Setup

For all the projects in this repository, you will need to create the following Conda environment in the first place:

```shell script
$ conda create -n drlnd python=3.6  # 3.7 does not work because of the version of Tensorflow!
$ conda activate drlnd
$ cd python
$ pip install .
```

Install Jupyter notebook:

```shell script
$ conda install jupyter
$ python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

Then you can select the kernel "drlnd" after starting a notebook.
