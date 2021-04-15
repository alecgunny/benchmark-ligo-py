#! /bin/bash -e

apt-get update
apt-get install wget git

wget -O ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ~/miniconda.sh -b -p $HOME/miniconda

export PATH="$HOME/miniconda/bin:$PATH"
conda init
conda update -n base conda

git clone https://github.com/alecgunny/benchmark-ligo-py
cd benchmark-ligo-py
conda create -f environment.yaml