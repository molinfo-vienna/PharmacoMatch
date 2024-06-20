import pandas as pd
import numpy as np
import argparse


def remove_duplicates(path_in, molecule_name_exists):
    # remove duplicate SMILES strings
    if int(molecule_name_exists) == 1:
        df = pd.read_csv(path_in, delimiter=" ", names=["SMILES", "ID"])
    else:
        df = pd.read_csv(path_in, names=["SMILES"])
    n_rows = len(df)
    df = df.drop_duplicates(subset="SMILES", keep="last")
    print(f"{n_rows-len(df)} duplicates were removed.")
    np.savetxt(path_in, df, fmt="%s", delimiter=" ")


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
        "-n",
        dest="molecule_name_exists",
        required=False,
        default=1,
        metavar="<molecule name exists>",
        help="Does a column for molecule names exists?",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parseArgs()  # process command line arguments
    remove_duplicates(args.in_file, args.molecule_name_exists)
