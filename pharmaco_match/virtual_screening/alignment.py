import sys
import os
import time

import torch

import CDPL.Pharm as Pharm
from dataset import VirtualScreeningDataModule


class PharmacophoreAlignment:
    """A class for aligning preprocessed ligands to a query pharmacophore.

    This class uses functions from the CDPL library to align preprocessed ligands to a
    query pharmacophore. The alignment scores are saved to a file in the dataset
    directory and can be loaded later for comparison with the matching scores of
    the PharmacoMatch model.

    Args:
        vs_root (str): Path to the root directory of the virtual screening dataset.
    """

    def __init__(self, vs_root: str) -> None:
        self.vs_root = vs_root

    def align_preprocessed_ligands_to_query(self) -> None:
        tic = time.perf_counter()
        self._alignment("actives")
        self._alignment("inactives")
        toc = time.perf_counter()
        self.alignment_time = toc - tic

    def _alignment(self, filename: str) -> None:
        ref_ph4_file = f"{self.vs_root}/raw/query.pml"
        in_file = f"{self.vs_root}/raw/{filename}.psd"
        out_file = f"{self.vs_root}/vs/all_{filename}_aligned.pt"

        # read the reference pharmacophore
        ref_ph4 = self._readRefPharmacophore(ref_ph4_file)

        # create reader for input molecules (format specified by file extension)
        # pharm_reader = Pharm.PSDPharmacophoreReader(in_file)
        db_accessor = Pharm.PSDScreeningDBAccessor(in_file)
        mol_ph4 = Pharm.BasicPharmacophore()

        # create instance of class implementing the pharmacophore alignment algorithm
        almnt = Pharm.PharmacophoreAlignment(
            True
        )  # True = aligned features have to be within the tolerance spheres of the ref. features

        # clear feature orientation information
        self._clearFeatureOrientations(ref_ph4)

        almnt.addFeatures(
            ref_ph4, True
        )  # set reference features (True = first set = reference)
        almnt.performExhaustiveSearch(
            False
        )  # set minimum number of top. mapped feature pairs

        # create pharmacophore fit score calculator instance
        almnt_score = Pharm.PharmacophoreFitScore(
            match_cnt_weight=1.0, pos_match_weight=0.9, geom_match_weight=0.0
        )

        almnt_scores = []

        # read and process molecules one after the other until the end of input has been reached
        num_ph4s = db_accessor.getNumPharmacophores()

        try:
            for i in range(num_ph4s):
                db_accessor.getPharmacophore(i, mol_ph4)
                mol_idx = db_accessor.getMoleculeIndex(i)
                conf_idx = db_accessor.getConformationIndex(i)

                try:
                    if mol_ph4.getNumFeatures() == 0:
                        continue

                    # clear feature orientation information
                    self._clearFeatureOrientations(mol_ph4)

                    almnt.clearEntities(
                        False
                    )  # clear features of previously aligned pharmacophore
                    almnt.addFeatures(
                        mol_ph4, False
                    )  # specify features of the pharmacophore to align

                    almnt_solutions = []  # stores the found alignment solutions

                    while (
                        almnt.nextAlignment()
                    ):  # iterate over all alignment solutions that can be found
                        score = almnt_score(
                            ref_ph4, mol_ph4, almnt.getTransform()
                        )  # calculate alignment score
                        almnt_solutions.append(score)

                    almnt_solutions = sorted(almnt_solutions, reverse=True)

                    if len(almnt_solutions) == 0:
                        almnt_scores.append(
                            [
                                0,
                                0,
                                mol_ph4.getNumFeatures(),
                                mol_idx,
                                conf_idx,
                            ]
                        )

                    else:
                        solution = almnt_solutions[0]
                        almnt_scores.append(
                            [
                                int(solution),
                                solution % 1,
                                mol_ph4.getNumFeatures(),
                                mol_idx,
                                conf_idx,
                            ]
                        )

                except Exception as e:
                    sys.exit(
                        "Error: pharmacophore alignment of molecule %s failed: %s"
                        # % (mol_id, str(e))
                        % (0, str(e))
                    )

        except Exception as e:  # handle exception raised in case of severe read errors
            sys.exit("Error: reading input molecule failed: " + str(e))

        almnt_scores = torch.tensor(almnt_scores)
        torch.save(almnt_scores, out_file)
        print(f"Alignment of {almnt_scores.shape[0]} ph4s")

    def _readRefPharmacophore(self, filename: str) -> Pharm.Pharmacophore:
        """Read query pharmacophore from the given file"""
        reader = Pharm.PharmacophoreReader(filename)

        # create an instance of the default implementation of the Pharm.Pharmacophore interface
        ph4 = Pharm.BasicPharmacophore()

        try:
            if not reader.read(ph4):  # read reference pharmacophore
                sys.exit("Error: reading reference pharmacophore failed")

        except Exception as e:  # handle exception raised in case of severe read errors
            sys.exit("Error: reading reference pharmacophore failed: " + str(e))

        return ph4

    def _clearFeatureOrientations(self, ph4: Pharm.BasicPharmacophore) -> None:
        """remove feature orientation informations and set the feature geometry to
        Pharm.FeatureGeometry.SPHERE"""
        for ftr in ph4:
            Pharm.clearOrientation(ftr)
            Pharm.setGeometry(ftr, Pharm.FeatureGeometry.SPHERE)


class ClassicalVirtualScreener:
    """Handler for virtual screening with the CDPKit alignment score.

    Handling of the alignment scores of the virtual screening dataset, which are
    calculated with CDPKit alignment. If existent, the scores are loaded from the
    dataset directory. Otherwise, the ligands are aligned to the query pharmacophore
    and the scores are saved to the dataset directory.

    Args:
        datamodule (VirtualScreeningDataModule): The datamodule of the virtual screening
            dataset.
    """

    def __init__(self, datamodule: VirtualScreeningDataModule) -> None:
        root_vs_dir = os.path.join(datamodule.virtual_screening_data_dir, "vs")

        if not os.path.exists(root_vs_dir):
            os.mkdir(root_vs_dir)
        actives_path = os.path.join(root_vs_dir, "all_actives_aligned.pt")
        inactives_path = os.path.join(root_vs_dir, "all_inactives_aligned.pt")

        if not os.path.exists(actives_path) or not os.path.exists(inactives_path):
            alignment = PharmacophoreAlignment(datamodule.virtual_screening_data_dir)
            alignment.align_preprocessed_ligands_to_query()

        self.actives_data = torch.load(actives_path)
        self.inactives_data = torch.load(inactives_path)
        self.actives_alignment_score = self.actives_data[:, 0] + self.actives_data[:, 1]
        self.inactives_alignment_score = (
            self.inactives_data[:, 0] + self.inactives_data[:, 1]
        )
        num_features_query = datamodule.metadata.query["num_features"].sum()

        self.actives_matches = self.actives_alignment_score >= num_features_query
        self.inactives_matches = self.inactives_alignment_score >= num_features_query

        self.alignment_score = torch.cat(
            (self.actives_alignment_score, self.inactives_alignment_score)
        )

        self.matches = torch.cat((self.actives_matches, self.inactives_matches))
