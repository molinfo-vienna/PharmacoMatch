import argparse
import os
import sys

import CDPL.Chem as Chem
import CDPL.Base as Base

# path_train = (
#     "/data/shared/projects/PhectorDB/training_data/preprocessing/chembl_data_phys.smi"
# )
# path_train = "/data/shared/projects/PhectorDB/training_data/preprocessing/chembl_data_filtered.smi"
# path_train_filtered = "/data/shared/projects/PhectorDB/training_data/preprocessing/chembl_data_filtered2.smi"
# root_dir = "/data/shared/projects/PhectorDB/DUD-E"


def remove_duplicates_and_test_data(
    path_train: str, test_root_dir: str, path_train_filtered: str
) -> None:
    # Create hash table from input training data
    mol_reader = Chem.MoleculeReader(path_train)
    mol = Chem.BasicMolecule()
    Chem.setMultiConfImportParameter(mol_reader, False)
    i = -1
    training_dict = dict()
    idx = set()

    while True:
        try:
            while mol_reader.read(mol):
                i = i + 1
                Chem.calcBasicProperties(mol, False)
                hashCode = Chem.calcHashCode(
                    mol,
                    Chem.AtomPropertyFlag.TYPE,
                    Chem.BondPropertyFlag.ORDER | Chem.BondPropertyFlag.AROMATICITY,
                )
                # Exclude duplicate structures and enantiomers
                if hashCode in training_dict.keys():
                    idx.add(i)
                    continue
                training_dict[hashCode] = i

            break

        except Base.IOError as e:
            print(
                f" Error: reading molecule at index {mol_reader.getRecordIndex()} failed",
                file=sys.stderr,
            )

            mol_reader.setRecordIndex(mol_reader.getRecordIndex() + 1)

    # Go through test data and remove from training data
    Chem.setMultiConfImportParameter(mol_reader, False)
    files = ["actives", "inactives"]

    for folder in os.listdir(test_root_dir):
        for file in files:
            path = f"{test_root_dir}/{folder}/preprocessing/{file}_phys.smi"
            if not os.path.exists(path):
                continue
            mol_reader = Chem.MoleculeReader(path)
            i = 0
            while True:
                try:
                    while mol_reader.read(mol):
                        Chem.calcBasicProperties(mol, False)
                        hashCode = Chem.calcHashCode(
                            mol,
                            Chem.AtomPropertyFlag.TYPE,
                            Chem.BondPropertyFlag.ORDER
                            | Chem.BondPropertyFlag.AROMATICITY,
                        )

                        if hashCode in training_dict.keys():
                            idx.add(training_dict[hashCode])

                        i = i + 1

                    break

                except Base.IOError as e:
                    print(
                        f" Error: reading molecule at index {mol_reader.getRecordIndex()} failed",
                        file=sys.stderr,
                    )

                    mol_reader.setRecordIndex(mol_reader.getRecordIndex() + 1)

    idx = sorted(list(idx))

    with open(path_train, "r") as f:
        lines = f.readlines()

    # rewrite the training data without the excluded smiles strings
    with open(path_train_filtered, "w") as f:
        for i, line in enumerate(lines):
            if i in idx:
                continue
            f.write(line)


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Removes duplicate strings from a SMILES file."
    )

    parser.add_argument(
        "-i",
        dest="in_file",
        required=True,
        metavar="<file>",
        help="Input molecule file",
    )

    parser.add_argument(
        "-o",
        dest="out_file",
        required=True,
        metavar="<file>",
        help="Output molecule file",
    )

    parser.add_argument(
        "-t",
        dest="test_dir",
        required=True,
        metavar="<file>",
        help="Root directory of test data",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parseArgs()  # process command line arguments
    remove_duplicates_and_test_data(args.in_file, args.test_dir, args.out_file)
