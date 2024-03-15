import os
import sys
from itertools import permutations

import CDPL.Pharm as Pharm


class PharmacophoreSubsetGenerator:
    def _getReaderByFileExt(self, filename: str) -> Pharm.PharmacophoreReader:
        name_and_ext = os.path.splitext(filename)

        if name_and_ext[1] == "":
            sys.exit(
                "Error: could not determine pharmacophore input file format (file extension missing)"
            )

        # get input handler for the format specified by the input file's extension
        ipt_handler = Pharm.PharmacophoreIOManager.getInputHandlerByFileExtension(
            name_and_ext[1][1:].lower()
        )

        if not ipt_handler:
            sys.exit(
                "Error: unsupported pharmacophore input file format '%s'"
                % name_and_ext[1]
            )

        # create and return file reader instance
        return ipt_handler.createReader(filename)

    def _getPharmWriterByFileExt(self, filename: str) -> Pharm.FeatureContainerWriter:
        name_and_ext = os.path.splitext(filename)

        if name_and_ext[1] == "":
            sys.exit(
                "Error: could not determine pharmacophore output file format (file extension missing)"
            )

        # get output handler for the format specified by the output file's extension
        opt_handler = Pharm.FeatureContainerIOManager.getOutputHandlerByFileExtension(
            name_and_ext[1][1:].lower()
        )

        if not opt_handler:
            sys.exit(
                "Error: unsupported pharmacophore output file format '%s'"
                % name_and_ext[1]
            )

        # create and return file writer instance
        return opt_handler.createWriter(filename)

    def generateSubsets(self, in_path, out_path):
        reader = self._getReaderByFileExt(in_path)
        writer = self._getPharmWriterByFileExt(out_path)
        ph4_subset = Pharm.BasicPharmacophore()
        ph4 = Pharm.BasicPharmacophore()
        min_num_features_subset = 6

        while reader.read(ph4):
            try:
                num_features = ph4.getNumFeatures()
                for i in range(num_features - min_num_features_subset + 1):
                    permutations_list = list(permutations(range(num_features), i))
                    for permutation in permutations_list:
                        ph4_subset = Pharm.BasicPharmacophore(ph4)
                        for feature in sorted(permutation, reverse=True):
                            ph4_subset.removeFeature(feature)
                        writer.write(ph4_subset)
            except Exception as e:
                sys.exit("Error: processing of pharmacophore failed: " + str(e))


if __name__ == "__main__":
    in_path = "/home/drose/git/PhectorDB/cdpkit_preprocessing/query_sub.pml"
    out_path = "/home/drose/git/PhectorDB/cdpkit_preprocessing/queries.pml"
    generator = PharmacophoreSubsetGenerator()
    generator.generateSubsets(in_path, out_path)
