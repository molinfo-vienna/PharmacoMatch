import argparse

import pandas as pd


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extracts the Smiles column from a CSV file and writes it to a new file."
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

    return parser.parse_args()


def csv2smi(path_in: str, path_out: str):
    df = pd.read_csv(path_in, delimiter=";")
    df = df["Smiles"]
    df.to_csv(path_out, index=False, header=False)


if __name__ == "__main__":
    args = parseArgs()
    csv2smi(args.in_file, args.out_file)
