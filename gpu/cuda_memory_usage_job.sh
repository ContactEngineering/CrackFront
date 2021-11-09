#MSUB -l walltime=00:15:00
#MSUB -m ea                                                                    
#MSUB -q gpu
#MSUB -l nodes=1:ppn=1:gpus=1 
#MSUB -l pmem=20gb


set -e

ml devel/cuda/11.3 
ml tools/singularity/3.5

PATH=$HOME/commandline:$PATH


WS=/work/ws/nemo/fr_as1412-2110_cf_gpu-0
IMAGE=$WS/CrackFront_gpu.sif

cd $MOAB_SUBMITDIR

# we expect that this script was copied in dataset/data and submitted from there

export KMP_AFFINITY=compact,1,0
export OMP_NUM_THREADS=1
# Add local, up to date CrackFront installation on top of path.
export PYTHONPATH=$WS/CrackFront:$PYTHONPATH

FILE=$WS/CrackFront/gpu/cuda_memory_usage.py


# --nv: for GPU
# --home=PWD: for jupyter

#singularity exec --nv --home=$WS --pwd $PWD -B $WS $IMAGE  \
#   jupytext --to notebook --output - $FILE | \
#   singularity exec --nv --home=$WS --pwd $PWD -B $WS $IMAGE \ 
#   jupyter nbconvert --execute --allow-errors -y --stdin --to=html --output=${FILE%*.py}.html
#singularity exec --nv --home=$WS --pwd $PWD -B $WS $IMAGE python3 $FILE

#echo "SUCCESS PYTHON"

singularity exec --nv --home=$WS --pwd $PWD -B $WS $IMAGE sh $WS/commandline/jupytext_to_html $FILE

echo "SUCCESS jupytext"