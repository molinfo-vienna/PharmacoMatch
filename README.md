# PhectorDB - Contrastive Learning of Pharmacophore Representations

## Abstract

The 3D-Pharmacophore is a common data structure in computational drug discovery to represent the key interactions between a ligand and a protein. Pharmacophores can be used to screen molecular databases for potential hit candidates. Traditional screening softwares like LigandScout work internally with an alignment algorithm to calculate a fit-score that is used to rank the database molecules w.r.t. some query pharmacophore. Although prefilter-methods exist, this process can be time-consuming.

This project aims to train an encoder model via a (self-supervised) contrastive loss to generate meaningful pharmacophore representations. The learned representation should encapsulate the spatial location and type of the pharmacophore features, and the similarity between two embedding vectors should correlate with the set similarity of the respective pharmacophores. Preprocessing a conformational database with this machine-learned representation should enable a fast similarity search with a given pharmacophore query. It could also be useful as a pre-filter for classical pharmacophore screening.

The method should be validated on several benchmark datasets and compared with existing methods for similarity search based on the enrichment factor of the obtained hit list. Baselines could be produced with existing molecular and pharmacophore fingerprints. The approach might also give comparable results to existing shape-based pharmacophore alignment techniques that are based on the volume overlap of the features.

## Getting started

**General**

Make sure that you have read the entries in our SSL Mattermost channel - this should give you a first start. I would also recommend to read through the material on our group wiki. 

You should have received an account from Thomas that enables you to access our intranet. We have several servers that are equipped with GPUs (srv3/4/5 and hydra). Make sure that you communciate with other group members if you intend to you use GPUs for a longer period of time (currently, this concerns mainly Sara). 

**Setting up the environment**



1. Install conda in the following directory `/data/shared/software/conda/<your name>` and set up a new conda environment

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

4. Make sure that the path to the nightly-build of the CDPKit is added to the `.bashrc` in your `Home` folder by pasting the following line at the end of the *>>>conda initialize>>>* block

```
export PYTHONPATH="${PYTHONPATH}:/data/shared/software/CDPKit-head-RH7/Python"
```

**Dataset location**

The data for this project can be found under `/data/shared/projects/PhectorDB/`.

The `training_data` folder contains:
- a dataset of approx. 1.2 M molecules as smiles  in the `input` folder. The molecules where compiled from the ChEMBL database and are used as unlabeled pretraining data. 
- the `preprocessing` folder contains the cleaned smiles after preprocessing, and their 3D-conformations as `.sdf`-files
- the `raw` folder contains the corresponding pharmacophores in `.cdf`-format
- `processed` contains the pytorch dataset saved to disk

[LIT-PCBA](https://drugdesign.unistra.fr/LIT-PCBA/) is a benchmark dataset. The `litpcba` folder currently contains one of the 15 different targets (ESR1_antago)
- Here, the `input` folder contains two files, `actives.smi` and `inactives.smi`.
- the `preprocessing` folder was created analogous to before
- The `pdb` folder contains one ligand-protein complex of the dataset's target. CDPKit works with the `.pdb` file for the protein and an `.sdf` file for the ligand
- `raw` contains the actives and inactives in `.psd` format, the pharmacophore screening database format of CDPKit. The `query.pml` file contains the interaction pharmacophore that was generated from the given pdb-structure
- `vs` contains the results of the pharmacophore alignment score as PyTorch tensors
- `processed` again contains the pytorch dataset saved to disk

**VS Code extensions**

I am currently using the Black formatter, isort, and the Flake8 linter, I would ask you to do the same to stay consistent throughout the project. GitHub Copilot is free for students, in case you want to try it out.

## TODOS

*Feel free to open an issue, if you are planning to tackle one of the problems below, we can discuss the details there. Make sure to create a new branch, when you start coding. Open a PR when you are finished and request a review, s.t. we can go through the pending changes.*

- [ ] The SimCLR loss function does not work with a continuous similarity measure between two augmented pharmacophores, it only declares them as similar or dissimilar. Incorporating the Jaccard similarity in the loss should enhance the model training, but as far as I understood this, the NTXent loss does not support this. I would like to try out two different approaches here. The first one would be to use a continuous contrastive loss function like they did in the [Vallina paper](https://arxiv.org/abs/2103.06638). Another idea could be to treat the model training as regression problem and use a MAE or MSE loss. Something [comparable](https://www.biorxiv.org/content/10.1101/2023.11.17.567506v1.full.pdf) was done a few months ago with the Schrödinger pharmacophore alignment score.

- [ ] It would be nice to investigate different pharmacophore encoders. So far, I tried a complete graph in combination with a GAT-layer and the PointTransformer architecture from the PyG tutorials. 

- [ ] Model Validation: Previously, I performed validation by comparing the similarity of a pharmacophore query embedding and the database ligand embeddings with the precomputed alignment score of a virtual screening run with the CDPKit. I would change this validation approach for several reasons, although comparable experiments should be done in the benchmarking section. VS is highly dependent on the selected query and dataset, but more importantly, the two algorithms do not achieve the same thing. The CDPKit VS is a substructure search, but the PhectorDB approach is a similarity search, so we should treat them differently and not compare them directly. I would store the augmented pharmacophores on disc instead of creating them on the fly and precompute the Jaccard-Similarity between the pairs. The model validation will simply be the similarity estimation on the validation set.

- [ ] I have not found a contrastive loss to enable substructure search, but if it exists, it would be definitely interesting, since this could enable actual VS in the classical sense. 

- [ ] SimCLR training took a long time, but training with the contrastive loss should go faster. If so, then it is time to implement an automated model optimization pipeline (grid search, random, or baysian).

- [ ] Model logging is currently done locally with the Lightning-logger on `data/shared`, but I heard a lot about the MLOps tool MLFlow. It would be interesting to see how we could use this for this project. It seems to be useful for project collaborations.

- [ ] When we are happy with a model that predicts pharmacophore similarity with satisfactory accuracy, it will be time for benchmarks. Now we will need to create pharmacophore queries that should achieve an enrichment that is better than random and compare it to other pharmacophore fingerprints that are used for similarity search. We might also compare the method here to the CDPKit algo, but I am not sure wether we should do that for the above mentioned reasons. We could include a runtime benchmark to show how fast similarity search can be done with a database like Zinc (or maybe Enamine, if we want to go for screening of very large data).

- [ ] Visualization are always good to have, and I had already some nice plots with the UMAP algo. This could also be compared to PCA (or MDS in the cosine similarity case).

- [ ] An ablation study would give us information on the importance of specific design decision. 

- [ ] The model accuracy w.r.t. the amount of training data will be of interest. Since we are creating our data artificially, this should be done rather quick. Another typical experiment is the accuracy w.r.t. the number of model parameters.

*Already achieved milestones:*

- [x] Compilation of a training dataset 
- [x] Data processing pipeline based on CDPKit scripts
- [x] Implementation of a pharmacophore dataset with the PyG dataset API logic and the Pytorch Lightning Data Module design pattern
- [x] Implementation of a pharmacophore encoder via a complete graph representation in PyG
- [x] Implementation of a contrastive learning framework (SimCLR) with Pytorch Lightning
- [x] Design and implementation of several pharmacophore augmentation strategies by subclassing of the PyG `BaseTransform`
- [x] Training monitoring with Tensorboard, Lightning Callbacks, and custom validation metrics
- [x] A Virtual Screening class for similarity search with the learned embeddings and a Jupyter Notebook for experimentation and data exploration
- [x] A self-similarity evaluation script