## PharmacoMatch: Efficient 3D Pharmacophore Screening through Neural Subgraph Matching


**General**

Due to the strict upload limit of 50 MB on OpenReview in the double-blind review process, we are unfortunately not able to provide you with a pretrained model or preprocessed data. We would therefore like to provide you with a detailed instruction on how to reproduce our results. Upon acceptance, we will make our GitHub repository publicly available, together with the preprocessed datasets via the Zenodo platform for data sharing.

**Hardware requirements**

We trained our model on a NVIDIA GeForce 3090 RTX graphics unit with 24 GB GDDR6X and strongly recommend GPU acceleration for model training. We further used an AMD EPYC 7713 64-Core Processor for data preprocessing and pharmacophore alignment.

**Setting up the environment**

1. Install conda and set up a new conda environment:

```
conda create -n <new_env> python==3.10.12
```

2. Now install the dependencies from the `requirements.txt` file. 

```
pip install -r requirements.txt
```

3. We need some of the additional PyG dependencies

```
pip install torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.0.1+cu117.html
```

4. Make sure to add the path to the `pharmaco_match` folder to your PYTHONPATH variable

5. For data processing, we are using the open-source software [CDPKit](https://cdpkit.org/index.html).
You can find installation instructions [here](https://cdpkit.org/installation.html) and installers for Linux/Mac/Windows [here](https://github.com/molinfo-vienna/CDPKit/releases). After installation, you will need to add the absolute path to the `Bin` folder to your PATH variable and the path to the `Python` folder to your PYTHONPATH variable. The `Bin` folder contains a compiled version of the conformer generation tool `confgen` and the `psdcreate` tool for the creation of pharmacophore databases from ligand datasets. The `Python` folder contains the `CDPL` module, which we will need for further data processing functionalities.

**Repository Contents**

Our project is structured as follows: 

- `data`: This folder contains unlabeled data for model training and virtual screening datasets for model evaluation.
- `data_preprocessing`: The subfolder `bash_scripts` contains `.sh`-files, which implement our data processing pipeline. The subfolder `python_scripts/cdpl` contains scripts for data processing, which were downloaded from the CDPKit homepage. You could alternatively download these files [here](https://cdpkit.org/cdpl_python_cookbook/index.html). The subfolder `python_scripts/utils` contains additional script that were used for preprocessing.
- `notebooks`: Contains a jupyter notebook for demonstration purposes.
- `pharmaco_match`: Contains our PyTorch code for model training. We further structured our code into a `dataset` subfolder for everything data related, `model` contains the implementation and training logic of our model, `scripts` contains the script for training, benchmarking, and further experiments, and `virtual screening` implements classes for virtual screening with our model. 

**Unlabeled Data Download**

Unlabeled data for this project was downloaded from the [ChEMBL database](https://www.ebi.ac.uk/chembl/web_components/explore/compounds/STATE_ID:iFvSIzcFcWFTVI47whwpSA%3D%3D). The website allows several filter options. We had set "Type: Small molecule" and "RO5 Violations: 0", which should result in approximately 1.34 M molecules. The data can be downloaded as `.csv`-file. 

We will only need the SMILES strings for our data processing pipeline. Copy the `SMILES` column without header and paste it into into the file `data/training_data/input/chembl_data.smi`. You could also use a utility file for this:

```python3 data_processing/python_scripts/utils/csv2smi.py -i <path to csv-file> -o data/training_data/input/chembl_data.smi```

**Unlabeled Data Preprocessing**

The 

The `data/training_data` folder contains:
- a dataset of approx. 1.2 M molecules as smiles  in the `input` folder. The molecules where compiled from the ChEMBL database and are used as unlabeled pretraining data. 
- the `preprocessing` folder contains the cleaned smiles after preprocessing, and their 3D-conformations as `.sdf`-files
- the `raw` folder contains the corresponding pharmacophores in `.cdf`-format
- `processed` contains the pytorch dataset saved to disk

**Model training**
Now that we have prepared the training data, we can finally train our model.
We can do it like this: ...

If your hardware does not meet the memory requirements, you will have to reduce the model size.
This could be easiest done by reducing the hidden layers of the convolutional layers in the config.file.
This will reduce the model performance. 

**Test datasets preprocessing**

We need to preprocess the test data for model evaluation. We can do it like this.

 The `data/DUD-E` folder contains ten different targets of the [DUD-E](https://dude.docking.org/) benchmark dataset. Since the files were not too big, we could include the input files and the pharamcophore query that was generated from the receptor structure.
- Here, the `input` folder contains two files, `actives.smi` and `inactives.smi`.
- the `preprocessing` folder was created analogous to before
- The `pdb` folder contains one ligand-protein complex of the dataset's target. CDPKit works with the `.pdb` file for the protein and an `.sdf` file for the ligand
- `raw` contains the actives and inactives in `.psd` format, the pharmacophore screening database format of CDPKit. The `query.pml` file contains the interaction pharmacophore that was generated from the given pdb-structure
- `vs` contains the results of the pharmacophore alignment score as PyTorch tensors
- `processed` again contains the pytorch dataset saved to disk

**Model Evaluation**

Now that we have the test datasets available, we can use them by execution the following scripts:

- `benchmark.py`
- `positional_perception.py`
- notebook


