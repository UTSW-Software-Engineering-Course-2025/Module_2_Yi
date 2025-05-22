# install Day 1 environment into jupyter python
module load python
conda activate /project/nanocourse/SWE_OOP/shared/CondaEnvs/
module unload python
python -m ipykernel install --name montillo_conda_env --user --display-name "montillo_conda_env"
module load python
conda deactivate
module unload python
