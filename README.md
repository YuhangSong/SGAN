# SGAN

This repo provides code for all the results reported in SGAN paper.

## Requirements

### Pre-requirements

Refer to [my personal basic setup](https://github.com/YuhangSong/basic_setup) for some convinient command lines.

* [CUDA 8.0](https://developer.nvidia.com/cuda-downloads)
* [Anaconda3](https://www.anaconda.com/download/)

### Other requirements

This process will requite root password, but note that we only use root permission fot apt-get install, so it is totally safe for your computer.

If you meet any failure, you are wellcome to report and pull request a fix, since we are trying to make our code usable as widely.

In order to install other requirements, follow:
```
# some install
sudo apt autoremove
sudo apt-get install -y tmux htop cmake golang libjpeg-dev git

# clean env
source ~/.bashrc
source deactivate
conda remove --name sgan_env --all

# create env
source ~/.bashrc
source deactivate
conda create -n sgan_env python=2 -y

# source in env
source ~/.bashrc
source activate sgan_env

# install
export CMAKE_PREFIX_PATH=~/anaconda3/
conda install -y numpy pyyaml mkl setuptools cmake gcc cffi numpy pyyaml mkl setuptools cmake gcc cffi
conda install -c soumith magma-cuda80

# clean dir
rm -r sgan_env

# create dir and install
mkdir -p sgan_env/project/
cd sgan_env/project/
wget http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp27-cp27mu-manylinux1_x86_64.whl
pip install http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp27-cp27mu-manylinux1_x86_64.whl
pip install numpy scipy torchvision universe six visdom "gym[atari]" matplotlib dill pygame imageio opencv-python
git clone https://github.com/YuhangSong/SGAN.git
cd SGAN
```

Install Pytorch from source (This has been depreciated, since the official release from Pytorch has supported all the features we need.)
```
git clone https://github.com/pytorch/pytorch.git && cd pytorch && git submodule update --init && python setup.py install && cd ..
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
