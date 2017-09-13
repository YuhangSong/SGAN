# SGAN
This repo provide code for all the result reported in SGAN paper.

## Setup enviroment
Vertual env on conda named sgan_env.
### Basic
Some basic setups for your computer, if you are familiar with these, pass them.
```
sudo apt-get install openssh-server
```
```
wget https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda_8.0.61_375.26_linux-run && sudo sh cuda_8.0.61_375.26_linux.run
```
```
wget https://repo.continuum.io/archive/Anaconda3-4.4.0-Linux-x86_64.sh && bash Anaconda3-4.4.0-Linux-x86_64.sh
```
### Setup Env
Setup env formally.
```
mkdir -p sgan_env/project/ && cd sgan_env/project/ && git clone https://github.com/YuhangSong/grl.git && sudo apt-get install -y tmux htop cmake golang libjpeg-dev && conda create -n sgan_env python=2 -y && source ~/.bashrc && source activate sgan_env && export CMAKE_PREFIX_PATH=~/anaconda3/ && conda install -y numpy pyyaml mkl setuptools cmake gcc cffi && conda install -c soumith magma-cuda80 && git clone https://github.com/pytorch/pytorch.git && cd pytorch && python setup.py install && pip install torchvision && pip install "gym[atari]" && pip install universe && pip install six && conda install -y numpy && conda install -y scipy && pip install scipy && pip install visdom && pip install matplotlib && pip install visdom && pip install sklearn && pip install dill && git config --global push.default "current" && git config --global pull.default "current" && git config --global credential.helper "cache --timeout=36000000000000000" && pip install pygame && pip install imageio && pip install opencv-python
```

## Run the code
We auto restore the checkpoint of the model, as well as restore the every plot from last run. If you want to start a new run, you should change the EXP parameter in main.py
```
source ~/.bashrc && source activate sgan_env && python main.py
```

## Visualize result
We store all results on disk, can be found in ../../result/
We also use visdom to visualize results remotely with visdom, so that you can observe your result remotely. The result you see in remote visdom will be same as what you have on the disk.

Start disdom server
```
source ~/.bashrc && source activate sgan_env && python -m visdom.server
```

Start ngrok if you the computer running this is behind a firewall
```
wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip && unzip ngrok-stable-linux-amd64.zip && ./ngrok http 8097
```

If you are using ngrok, browse the address provided by above command, like: http://bbc9b540.ngrok.io/. If you do not use ngrok, browse: <your ip>:8097

## Note
Note that the default program run SGAN on 2D Grid with uniform action dynamic. To ran baseline of GP-WGAN, change the params of METHOD. To run other baselines and domains, give all the add_parameter() a look, they are simple and straight forward. Note that we latter found the baseline of GP-WGAN could be even worse than we reported in paper. It tends to stack at a very bad point quite often, so if you get a result of L1=1.8 and AR=0.16, something like that, do not be surprise. But we do not see this unstable in our SGAN for now.
