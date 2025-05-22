# Setup GPU based on training account (even: 0; odd: 1)
# Ex: train00
export TF_CPP_MIN_LOG_LEVEL=2
export TF_USE_LEGACY_KERAS=1
export CUDA_VISIBLE_DEVICES=$(($(whoami | grep -Eo '[1-9][0-9]*')%2))
echo "My assigned GPU is card: $CUDA_VISIBLE_DEVICES"

# Conda Env
module load python
conda activate /project/nanocourse/SWE_OOP/shared/CondaEnvs/
module unload python

