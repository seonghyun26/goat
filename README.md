# [Collective Variable Free Transition Path Sampling with Generative Flow Networks](https://arxiv.org/abs/2405.19961v2)

## Environment settings

This code runs on version python version: 3.9

Install the followings:

- Openmmtools for Molecular Dynamics (MD) simulation 
    ```
    conda install -c conda-forge openmmtools
    ```
- Openmmforcefields for forcefields of large proteins
    ```
    git clone https://github.com/openmm/openmmforcefields.git
    ```
- Other packages
    ```
    pip install torch tqdm wandb mdtraj matplotlib
    ```

## Usage

- **Training**: Run the following command to start training:
    ```
    bash scripts/train_alanine.sh
    ```

- **Evaluation**: Run the following command to perform evaluation:
    ```
    bash scripts/eval_alanine.sh
    ```

## Reproduce

- **Table**: Run the following command to reproduce the table:
    ```
    bash scripts/reproduce_table.sh
    ```
