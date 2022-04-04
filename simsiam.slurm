#!/bin/bash
#SBATCH --job-name=SimSiam     # job name
#SBATCH --ntasks=4                   # number of MP tasks
#SBATCH --ntasks-per-node=4          # number of MPI tasks per node
#SBATCH --gres=gpu:4                 # number of GPUs per node
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --distribution=block:block   # we pin the tasks on contiguous cores
#SBATCH --time=00:03:00              # maximum execution time (HH:MM:SS)
#SBATCH --qos=qos_gpu-t3
#SBATCH --output=logs/%A_%a.out # output file name
#SBATCH --error=logs/%A_%a.err  # error file name
#SBATCH --array=0


set -x
cd $WORK/simsiam

module purge
module load pytorch-gpu/py3/1.8.1

export TORCH_HOME=${SCRATCH}/pretrained_models
export TQDM_DISABLE='yes'
export TMPDIR=$JOBSCRATCH


# 'experience.experiment_name=${loss.name}_${dataset.sampler.kwargs.batch_size}'
loss[0]=supap

srun python main_simsiam.py \
-a resnet50 \
--dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
--fix-pred-lr \
${SCRATCH}/imagenet
