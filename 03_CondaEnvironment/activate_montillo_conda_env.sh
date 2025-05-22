export TF_CPP_MIN_LOG_LEVEL=2
export TF_USE_LEGACY_KERAS=1

### Use this with training accounts
# Setup GPU based onining account (even: 0; odd: 1)
account=$(whoami | grep -Eo '[1-9][0-9]*')
if [ ! -z $account ];
then 
	export CUDA_VISIBLE_DEVICES=$(($account%2))
    echo "Assigned GPU: $CUDA_VISIBLE_DEVICES"
else
	echo "Unable to set CUDA_VISIBLE_DEVICES from training account. Setting to 0"
	export CUDA_VISIBLE_DEVICES=0	
fi

# install Day 1 environment into jupyter python
module load python
conda activate /project/nanocourse/SWE_OOP/shared/CondaEnvs/
module unload python
