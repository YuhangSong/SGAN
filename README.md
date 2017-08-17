# GRL

# Quik setup enviroment: grl_2
```
sudo apt-get install openssh-server
```
```
wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip && unzip ngrok-stable-linux-amd64.zip
```
```
wget https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda_8.0.61_375.26_linux-run && sudo sh cuda_8.0.61_375.26_linux.run && wget https://repo.continuum.io/archive/Anaconda3-4.4.0-Linux-x86_64.sh && bash Anaconda3-4.4.0-Linux-x86_64.sh && mkdir -p YOUR_NAME/project/ && cd YOUR_NAME/project/ && git clone https://github.com/YuhangSong/grl.git && sudo apt-get install -y tmux htop cmake golang libjpeg-dev && conda create -n grl_2 python=2 -y && source ~/.bashrc && source activate grl_2 && export CMAKE_PREFIX_PATH=~/anaconda3/ && conda install -y numpy pyyaml mkl setuptools cmake gcc cffi && conda install -c soumith magma-cuda80 && git clone https://github.com/pytorch/pytorch.git && cd pytorch && python setup.py install && pip install torchvision && pip install "gym[atari]" && pip install universe && pip install six && conda install -y numpy && conda install -y scipy && pip install scipy && pip install visdom && pip install matplotlib && pip install visdom && pip install sklearn && pip install dill && git config --global push.default "current" && git config --global pull.default "current" && git config --global credential.helper "cache --timeout=36000000000000000"
```
### Optinal
```
pip install opencv-python
```

# Train
#### We auto restore the checkpoint of the model, as well as restore the every plot from last run. If you want to start a new run, you should change the EXP in rgan.py
```
source ~/.bashrc && source activate grl_2 && python rgan.py
```

# Visualize result
#### We store all results on disk, can be found in ../../result/
#### We also use visdom to visualize results remotely (optianal)
##### Start disdom server
```
source ~/.bashrc && source activate grl_2 && python -m visdom.server
```
##### Start ngrok so that we can vist visdom page remotely
```
wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip && unzip ngrok-stable-linux-amd64.zip && ./ngrok http 8097
```
##### Brows the address provided by above command, like: http://bbc9b540.ngrok.io/
