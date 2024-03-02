# Pose Estimation 

## Local Setup

Setup conda in wsl
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
conda --version
```

### Setup env with conda
```bash
cd runtime
conda env create -f environment.yml
conda activate <env> 
```

## Common Issues
Issue #1
```
ImportError: libGL.so.1: cannot open shared object file: No such file or directory
```
Solution #1
```
sudo apt-get install libgl1-mesa-glx
```


## General Commands
Prepare and test submissions:
```
$ make clean && make pack-final && make test-submission
```

```
$ pip install -r runtime/environment.yml 
```

Run docker in interactive mode:

`docker run -it spacecraftpose.azurecr.io/spacecraft-pose-pose-estimation
/bin/bash`
