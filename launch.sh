#!/bin/bash -l
###SBATCH --job-name=lite-hrnet-demo
###SBATCH --dependency singleton
#SBATCH --time=1-12:00:00
#SBATCH --partition=gpu # If gpu: set '-G <gpus>'
#SBATCH -G 1
#SBATCH -N 1 # Number of nodes
#SBATCH --ntasks-per-node=2
#SBATCH -c 1 # multithreading per task
#SBATCH -o %x-%j.out # <jobname>-<jobid>.out           

export MODULEPATH=/opt/apps/resif/iris/2019b/gpu/modules/all
module load lang/Python/3.7.4-GCCcore-8.3.0

source envs/lite_hrnet/bin/activate

cd sat-pose-estimation/Lite-HRNet

python -m tools.train configs/top_down/lite_hrnet/Envisat/litehrnet_18_coco_256x256_Envisat+IC.py
