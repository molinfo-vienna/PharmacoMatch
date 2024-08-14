### PharmacoMatch: Efficient 3D Pharmacophore Screening through Neural Subgraph Matching

**General**

This repository contains the code and data appendix for our AAAI-25 submission. Due to the 50 MB upload limit on OpenReview during the double-blind review process, we are unable to provide preprocessed data at this time. However, we have included detailed instructions on how to reproduce our results. Upon acceptance, the GitHub repository will be made public, along with the preprocessed datasets via the Zenodo platform. A pretrained model is readily available for use.

**System specifications**

- **GPU:** NVIDIA GeForce 3090 RTX with 24 GB GDDR6X (recommended for model inference and training).
- **CPU:** AMD EPYC 7713 64-Core Processor (used for data preprocessing and pharmacophore alignment).
- **OS:** Rocky Linux (v.9.4).

**Setting Up the Environment**

1. **Install Conda and create a new environment:**

    ```bash
    conda create -n pharmaco_match python==3.10.12
    conda activate pharmaco_match
    ```

2. **Navigate to the `PharmacoMatch` folder and install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Install additional PyG dependencies:**

    ```bash
    pip install torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.0.1+cu117.html
    ```

4. **Add the `pharmaco_match` folder to your `PYTHONPATH`:**

    Ensure that the path to the `pharmaco_match` folder is added to the `PYTHONPATH` variable in your Conda environment.

**Setting Up the Chemical Data Processing Toolkit (CDPKit)**

For data processing, we use the open-source software [CDPKit](https://cdpkit.org/index.html). Follow the installation instructions provided [here](https://cdpkit.org/installation.html), installers can be found [here](https://github.com/molinfo-vienna/CDPKit/releases). The installation will include a `CDPKit/Bin` subfolder containing command-line applications.

Key tools:
- `confgen`: Generates `.sdf` files from `.smi` files for conformer generation.
- `psdcreate`: Creates pharmacophore database files (`.psd` files) from `.sdf` files.

**Repository Contents**

- **`data`:** Contains unlabeled data for model training and virtual screening datasets for model evaluation.
- **`data_preprocessing`:** 
    - `bash_scripts`: Implements the data processing pipeline with `.sh` files.
    - `python_scripts/cdpl`: Contains scripts for data processing from the [CDPKit website](https://cdpkit.org/cdpl_python_cookbook/index.html).
    - `python_scripts/utils`: Contains additional preprocessing scripts.
- **`pharmaco_match`:** Our PyTorch code for model training, structured as:
    - `dataset`: Data-related modules.
    - `model`: Model implementation and training logic.
    - `scripts`: Training, benchmarking, and experimental scripts.
    - `virtual_screening`: Classes for virtual screening with our model.
- **`trained_model`:** Stores the pretrained model used in our study.

**IMPORTANT**

Edit the path variables at the top of the bash scripts to match your CDPKit installation and PharmacoMatch root-folder:

```bash
cdpkit='<your_path>/CDPKit/Bin'
project='<your_path>/PharmacoMatch'
```

**Test Datasets Processing**

The `data/DUD-E` folder contains ten different targets from the [DUD-E](https://dude.docking.org/) benchmark dataset. Since the files are not too large, we have included the input files and pharmacophore queries generated from the receptor structure. To run the data processing pipeline, execute:

```bash
./data_processing/bash_scripts/vs_data_generation.sh
```

The folder structure of the test sets should now include:
- **`input`:** Contains `actives.smi` and `inactives.smi` files.
- **`preprocessing`:** Contains intermediate processing files.
- **`raw`:** Includes actives and inactives in `.psd` format and `query.pml` for interaction pharmacophore.
- **`vs`:** Stores pharmacophore alignment scores.
- **`processed`:** Contains the final PyTorch datasets.

**Model Evaluation**

With the test datasets processed, you can now execute the following script:

- **`benchmark.py`:** Embeds test datasets, generates visualizations, and calculates evaluation metrics as reported in our paper.

This script requires processing of the training data, which we will explain in the next step:

- **`positional_perception.py`:** Performs the positional perception experiment. 

**Unlabeled Data for Model Training**

To train the model, download the unlabeled data from the [ChEMBL database](https://www.ebi.ac.uk/chembl/web_components/explore/compounds/STATE_ID:iFvSIzcFcWFTVI47whwpSA%3D%3D). Filter by "Type: Small molecule" and "RO5 Violations: 0" to obtain approximately 1.34 million molecules. 

Process the SMILES strings by pasting them into `data/training_data/input/chembl_data.smi` or use the utility:

```bash
python3 data_processing/python_scripts/utils/csv2smi.py -i <path to csv-file> -o data/training_data/input/chembl_data.smi
```

Run the preprocessing script:

```bash
./data_processing/bash_scripts/training_data.sh
```

**Model Training**

Once the training data is prepared, you can train the model:

```bash
python3 pharmaco_match/scripts/training.py 0
```

Where `0` corresponds to the GPU device index. If your hardware doesn't meet the memory requirements, reduce the model size by adjusting the hidden layers in the `config.yaml` file (note that this may reduce model performance).


