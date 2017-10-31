# SGAN

This repo provides code for all the results reported in SGAN paper. See [Learning
Approximate Stochastic Transition Models.](https://arxiv.org/abs/1710.09718)

## Requirements

### Pre-requirements

Follow [requirements from another repo of mine](https://github.com/YuhangSong/gtn_a2c#requirements) to setup a basic env.

### Other requirements

Install other requirements, follow:
```
# clear env
source ~/.bashrc
source deactivate
conda remove --name sgan_env --all
# create env
source ~/.bashrc
source deactivate
conda create -n sgan_env --clone gtn_env

# source in env
source ~/.bashrc
source activate sgan_env

# clean dir and create dir and install
sudo rm -r sgan_env 
mkdir -p sgan_env/project/
cd sgan_env/project/
pip install pddlpy

# done
git clone https://github.com/YuhangSong/SGAN.git
cd SGAN
```

## Run the code

The code auto restore the checkpoint of the model, as well as restore any plot from the last run.
If you want to start a new run without restore anything, you should change the EXP parameter in main.py
```
source ~/.bashrc
source activate sgan_env
python main.py
```

## Visualize result

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
