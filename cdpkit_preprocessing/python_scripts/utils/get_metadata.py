import sys
import os
import argparse

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

import CDPL.Chem as Chem
import CDPL.Pharm as Pharm


# returns a Chem.MoleculeReader instance for the specified molecule input file format
def getMolReaderByFileExt(filename: str) -> Chem.MoleculeReader:
    name_and_ext = os.path.splitext(filename)

    if name_and_ext[1] == "":
        sys.exit(
            "Error: could not determine molecule input file format (file extension missing)"
        )

    # get input handler for the format specified by the input file's extension
    ipt_handler = Chem.MoleculeIOManager.getInputHandlerByFileExtension(
        name_and_ext[1][1:].lower()
    )

    if not ipt_handler:
        sys.exit("Error: unsupported molecule input file format '%s'" % name_and_ext[1])

    # create and return file reader instance
    return ipt_handler.createReader(filename)


def get_alignment_metadata(file):
    path = f"/data/shared/projects/PhectorDB/virtual_screening_ESR1_ant/vs/{file}.sdf"
    reader = getMolReaderByFileExt(path)
    Chem.setMultiConfImportParameter(reader, False)
    mol = Chem.BasicMolecule()
    meta_data = []

    try:
        while reader.read(mol):
            struct_data = Chem.getStructureData(mol)  #
            row = dict()
            for (
                entry
            ) in (
                struct_data
            ):  # iterate of structure data entries consisting of a header line and the actual data
                row[entry.header] = entry.data

            meta_data.append(row)

    except Exception as e:  # handle exception raised in case of severe read errors
        sys.exit("Error: reading molecule failed: " + str(e))

    return pd.DataFrame(meta_data)


def get_best_query(df):
    hit_counts = df["<Query Pharm. Index>"].value_counts()
    pharmacophore_idx = hit_counts.idxmax()
    print(f"The ph4 {pharmacophore_idx} has {hit_counts.loc[pharmacophore_idx]} hits.")


def main() -> None:
    actives = get_alignment_metadata("actives_aligned")
    inactives = get_alignment_metadata("inactives_aligned")
    get_best_query(actives)
    inactive_scores = inactives["<Score>"].to_numpy("float32")
    active_scores = actives["<Score>"].to_numpy("float32")
    y_true = np.concatenate(
        (np.ones(len(active_scores)), np.zeros(len(inactive_scores)))
    )
    y_pred = np.concatenate((active_scores, inactive_scores))
    score = roc_auc_score(y_true, y_pred)
    print(f"The rocauc score is {score}")


if __name__ == "__main__":
    main()
