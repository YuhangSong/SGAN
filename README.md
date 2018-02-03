# SGAN

This repository provides **database** and **code** for reproducing all the results in:

* [**Learning
Approximate Stochastic Transition Models.**](https://arxiv.org/abs/1710.09718)
[*Yuhang Song*](https://yuhangsong.my.cam/),
[*Christopher Grimm*]().
[*Xianming Wang*]().
[*Michael L. Littman* &#8727;](http://cs.brown.edu/~mlittman/).
By [HCRI](https://hcri.brown.edu/) @ [Brown University](https://www.brown.edu/).

[Amidar](https://www.youtube.com/watch?v=3sJubQAXSUc)  |  [Alien](https://www.youtube.com/watch?v=bOZ7TIx5Zv8&t=47s)  |  [Assault](https://www.youtube.com/watch?v=HwWJrb2PQQ0&t=38s)
:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/YuhangSong/SGAN/blob/master/imgs/Amidar.gif)  |  ![](https://github.com/YuhangSong/SGAN/blob/master/imgs/Alien.gif)  |  ![](https://github.com/YuhangSong/SGAN/blob/master/imgs/Assault.gif)
[BeamRider](https://www.youtube.com/watch?v=dgTxca0pdac)  |  [Boxing](https://www.youtube.com/watch?v=Ev0_hmee2cM)  |  [DemonAttack](https://www.youtube.com/watch?v=p67oOM4rjcU&t=31s)
![](https://github.com/YuhangSong/SGAN/blob/master/imgs/BeamRider.gif)  |  ![](https://github.com/YuhangSong/SGAN/blob/master/imgs/Boxing.gif)  |  ![](https://github.com/YuhangSong/SGAN/blob/master/imgs/DemonAttack.gif)
[WizardOfWor](https://www.youtube.com/watch?v=1wFzsa-S_fY)  |  [MsPacman](https://www.youtube.com/watch?v=CYjSI4Pnh7M) |  [Phoenix](https://www.youtube.com/watch?v=3ILULkcBRG0)
![](https://github.com/YuhangSong/SGAN/blob/master/imgs/WizardOfWor.gif)  |  ![](https://github.com/YuhangSong/SGAN/blob/master/imgs/MsPacman.gif)  |  ![](https://github.com/YuhangSong/SGAN/blob/master/imgs/Phoenix.gif)

Specifically, this repository includes extremely simple guidelines to:
* Download and setup the Stochastic World database.
* Setup a friendly environment to run our code (including the part handling stochastic Atari games).

## Download and setup Stochastic World database

Our Stochastic World database contains two physical games of **Marble Game** and **Chaos Double Pendulum**. Each game was recorded for more than **24** hours with more then **10e7** frames.

![](https://github.com/YuhangSong/SGAN/blob/master/imgs/marble_all.gif)  |  ![](https://github.com/YuhangSong/SGAN/blob/master/imgs/marble_single.gif)
:-------------------------:|:-------------------------:|
![](https://github.com/YuhangSong/SGAN/blob/master/imgs/marble_all.gif)  |  ![](https://github.com/YuhangSong/SGAN/blob/master/imgs/marble_single.gif)

Follow command lines here to download and setup our PVS-HM database:
```
mkdir -p sgan_env/dataset/
cd sgan_env/dataset/
wget https://drive.google.com/open?id=0B20VnLepDOwfUFhHLVFDR2VjVDg
unzip dataset.zip
```

## Setup an environment to run our code

### Pre-requirements

* [CUDA](https://developer.nvidia.com/cuda-downloads)
* [Anaconda3](https://www.anaconda.com/download/) (Python 3.6)

If you are not familiar with above things, refer to [my personal basic setup](https://github.com/YuhangSong/Cool-Ubuntu-For-DL) for some guidelines.
The code should also be runnable without a GPU, but I would not recommend it.

### Requirements

There will be command lines after the list, you don't have to install below requirements one by one.
Besides, if you are not familiar with below things, I highly recommend you to just follow command lines after the list:
* Python 3.6
* [Pytorch](http://pytorch.org/)
* [torchvision](https://github.com/pytorch/vision)
* [numpy](http://www.numpy.org/)
* [gym](https://github.com/openai/gym)
* [imageio](https://imageio.github.io/)
* [matplotlib](https://matplotlib.org/)
* [pybullet](https://pypi.python.org/pypi/pybullet)
* [opencv-python](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_tutorials.html)

Install above requirements with command lines:
```
# create env
conda create -n grl_env

# source in env
source ~/.bashrc
source activate sgan_env

# install requirements
pip install http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp36-cp36m-manylinux1_x86_64.whl # if you are using CUDA 8.0, otherwise, refer to their official site: http://pytorch.org/
pip install torchvision
pip install visdom
pip install numpy -I
pip install gym[atari]
pip install imageio
pip install matplotlib
pip install pybullet
pip install opencv-python
pip install pddlpy

# clean dir and create dir
mkdir -p sgan_env/project/
cd sgan_env/project/
git clone https://github.com/YuhangSong/SGAN.git
cd SGAN
```

Meet some issues? See [problems](https://github.com/YuhangSong/GTN#problems). If there isn't a solution, please don not hesitate to open an issue.

## Run our code

#### Run SGAN.

The code auto restore the checkpoint of the model, as well as restore any plot from the last run.
If you want to start a new run without restore anything, you should change the EXP parameter in main.py

```bash
source ~/.bashrc
source activate sgan_env
CUDA_VISIBLE_DEVICES=0 python main.py
```

#### Run other baselines.
Give ```main.py``` a look, it is well commented.

#### Visualize result

Start a `Visdom` server with
```bash
source ~/.bashrc
source activate sgan_env
python -m visdom.server
```
Visdom will serve `http://localhost:8097/` by default.

We store all results on disk, can be found in ../../result/.
The result is located in a directory named by the parameters you set in the code. So, every time you specific any new parameter to a new value, the directory will change accordingly.

We also use visdom to visualize results remotely, so that you can observe your result from any browser (Recommend Google [Chrome](https://www.google.com/chrome/browser/desktop/index.html?brand=CHBD&gclid=Cj0KCQjwgb3OBRDNARIsAOyZbxDQqD8yexBYnNgpuh8Taiqzk0H_VCmNnYibw3SdWL7uqx0L3GOJicAaAkEFEALw_wcB)).
The result you see in the remote visdom will be same as what you have stored on the disk.

To enable remote visdom, start disdom server by running following commend in the local machine (The machine you run main.py).
Note that if you do not need remote visdom, we recommend you also start following visdom server, in order to avoid warning massage printing in the console.
```
source ~/.bashrc
source activate sgan_env
python -m visdom.server
```

## Note
Note that the default program run SGAN on 2D Grid with uniform action dynamic.
To ran baseline of GP-WGAN, change the params of METHOD.
To run other baselines and domains, give all the add_parameter() a look, they are simple and straight forward.
Note that we later found the baseline of GP-WGAN could be even worse than we reported in paper.
It tends to stack at a very bad point quite often (5 failures out of 10 runs).
So, if you get a result of L1=1.8 and AR=0.16 on GP-WGAN, something like that, do not be surprise.
But we do not see this unstable in our SGAN for now (out of 10 runs).
