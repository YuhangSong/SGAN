# GRL

# Quik setup enviroment: grl_1
```
mkdir -p YOUR_NAME/project/ && cd YOUR_NAME/project/ && git clone https://github.com/YuhangSong2017/grl.git && sudo apt-get install -y tmux htop cmake golang libjpeg-dev && ~/anaconda3/bin/conda create -n grl_1 python=2 -y && source ~/.bashrc && source activate grl_1 && export CMAKE_PREFIX_PATH=~/anaconda3/ && conda install -y numpy pyyaml mkl setuptools cmake gcc cffi && conda install -c soumith magma-cuda80 && git clone https://github.com/pytorch/pytorch.git && cd pytorch && python setup.py install && pip install torchvision && pip install tensorflow && pip install "gym[atari]" && pip install universe && pip install six && conda install -y numpy && conda install -y scipy && pip install opencv-python && pip install scipy && pip install visdom && pip install matplotlib && pip install visdom && pip install sklearn && pip install dill
```

# Train
#### We auto restore the checkpoint of the model, as well as restore the every plot from last run. If you want to start a new run, you should change the EXP in rgan.py
```
source ~/.bashrc && source activate grl_1 && python rgan.py
```

# Visualize result
#### We use visdom to visualize results remotely (optianal, we also store all results on disk, can be found in ../../result/)
##### Start disdom server
```
source activate grl_1 && python -m visdom.server
```
##### Start ngrok so that we can vist visdom page remotely
```
wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip && unzip ngrok-stable-linux-amd64.zip && ./ngrok http 8097
```
