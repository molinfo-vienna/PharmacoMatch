import pandas as pd


def csv2smi():
    # extract smiles col from csv file
    path = "/data/shared/projects/PhectorDB/chembl_data/chembl_data.csv"
    path_out = "/data/shared/projects/PhectorDB/chembl_data/chembl_data.smi"
    df = pd.read_csv(path, delimiter=";")
    df = df["Smiles"]
    df.to_csv(path_out, index=False, header=False)


def remove_duplicates():
    # remove duplicate SMILES strings
    path = "/data/shared/projects/PhectorDB/chembl_data/clean.smi"
    path_out = (
        "/data/shared/projects/PhectorDB/chembl_data/clean_without_duplicates.smi"
    )
    df = pd.read_csv(path)
    unique_smiles = list(set(df.values.flatten().tolist()))
    print(f"{len(df)-len(unique_smiles)} duplicates were removed.")
    df = pd.DataFrame(unique_smiles)
    df.to_csv(path_out, index=False, header=False)


remove_duplicates()
