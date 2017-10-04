# SGAN

This repo provides code for all the results reported in SGAN paper.

## Setup enviroment

Virtual environment on conda named sgan_env.

### Basic

Some basic setups for your computer, if you are familiar with these, just ignore them.

Install SSH for remote control.
```
sudo apt-get install openssh-server
```

Download and install [CUDA 8.0](https://developer.nvidia.com/cuda-downloads).
```
wget https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda_8.0.61_375.26_linux-run && sudo chmod -x cuda_8.0.61_375.26_linux-run && mv cuda_8.0.61_375.26_linux-run cuda_8.0.61_375.26_linux.run && sudo sh cuda_8.0.61_375.26_linux.run
```

Download and install [Anaconda3](https://www.anaconda.com/download/).
```
wget https://repo.continuum.io/archive/Anaconda3-4.4.0-Linux-x86_64.sh && bash Anaconda3-4.4.0-Linux-x86_64.sh
```

### Setup Env

Setup environment.
This process will requite root password, but note that we only use root permission fot apt-get install, so it is totally safe for your computer.
If you meet any failure, you are wellcome to report and pull request a fix, since we are trying to make our code usable as widely.
```
mkdir -p sgan_env/project/ && cd sgan_env/project/ && sudo apt-get install -y tmux htop cmake golang libjpeg-dev git && conda create -n sgan_env python=2 -y && source ~/.bashrc && source activate sgan_env && export CMAKE_PREFIX_PATH=~/anaconda3/ && conda install -y numpy pyyaml mkl setuptools cmake gcc cffi && conda install -c soumith magma-cuda80 && sudo apt install -y git && git clone https://github.com/pytorch/pytorch.git && cd pytorch && git submodule update --init && python setup.py install && pip install torchvision && pip install "gym[atari]" && pip install universe && pip install six && conda install -y numpy && conda install -y scipy && pip install scipy && pip install visdom && pip install matplotlib && pip install visdom && pip install sklearn && pip install dill && git config --global push.default "current" && git config --global pull.default "current" && git config --global credential.helper "cache --timeout=36000000000000000" && pip install pygame && pip install imageio && pip install opencv-python && git clone https://github.com/YuhangSong/SGAN.git && cd SGAN
```

## Run the code

The code auto restore the checkpoint of the model, as well as restore any plot from the last run.
If you want to start a new run without restore anything, you should change the EXP parameter in main.py
```
source ~/.bashrc && source activate sgan_env && python main.py
```

## Visualize result

We store all results on disk, can be found in ../../result/.
The result is located in a directory named by the parameters you set in the code. So, every time you specific any new parameter to a new value, the directory will change accordingly.

We also use visdom to visualize results remotely, so that you can observe your result from any browser (Recommend Google [Chrome](https://www.google.com/chrome/browser/desktop/index.html?brand=CHBD&gclid=Cj0KCQjwgb3OBRDNARIsAOyZbxDQqD8yexBYnNgpuh8Taiqzk0H_VCmNnYibw3SdWL7uqx0L3GOJicAaAkEFEALw_wcB)).
The result you see in the remote visdom will be same as what you have stored on the disk.

To enable remote visdom, start disdom server by running following commend in the local machine (The machine you run main.py).
Note that if you do not need remote visdom, we recommend you also start following visdom server, in order to avoid warning massage printing in the console.
```
source ~/.bashrc && source activate sgan_env && python -m visdom.server
```

If you the computer running this is behind a firewall ,start ngrok by running following commend in the local machine (The machine you run main.py).
```
wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip && unzip ngrok-stable-linux-amd64.zip && ./ngrok http 8097
```

If you do not use ngrok, browse: <your ip>:8097
If you are using ngrok, browse the address provided by above command, like: http://bbc9b540.ngrok.io/.

## Note
Note that the default program run SGAN on 2D Grid with uniform action dynamic.
To ran baseline of GP-WGAN, change the params of METHOD.
To run other baselines and domains, give all the add_parameter() a look, they are simple and straight forward.
Note that we later found the baseline of GP-WGAN could be even worse than we reported in paper.
It tends to stack at a very bad point quite often (5 failures out of 10 runs).
So, if you get a result of L1=1.8 and AR=0.16 on GP-WGAN, something like that, do not be surprise.
But we do not see this unstable in our SGAN for now (out of 10 runs).
