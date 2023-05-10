#! /usr/bin/env bash

#SBATCH --job-name=
#SBATCH --output=
#SBATCH --ntask=1
#SBATCH --mem-per-cpu=1
#SBATCH --partition=

cd $HOME
module purge
module load Python/Python3.9

if [ ! -d dynamic ] && git clone https://github.com/echonet/dynamic.git
cd dynamic
pip install --user -e .

cp ../EchoNet-Dynamic.zip && unzip EchoNet-Dynamic.zip
echo 'DATA_DIR = EchoNet-Dynamic/' > echonet.cfg

echonet segmentation --save_video