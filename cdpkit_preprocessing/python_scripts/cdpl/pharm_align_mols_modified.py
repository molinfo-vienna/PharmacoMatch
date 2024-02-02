#!/bin/env python

##
# pharm_align_mols.py
#
# This file is part of the Chemical Data Processing Toolkit
#
# Copyright (C) 2003 Thomas Seidel <thomas.seidel@univie.ac.at>
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2 of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program; see the file COPYING. If not, write to
# the Free Software Foundation, Inc., 59 Temple Place - Suite 330,
# Boston, MA 02111-1307, USA.
##


import sys
import time

import torch

import CDPL.Chem as Chem
import CDPL.Pharm as Pharm
import CDPL.Math as Math


# reads and returns the specified alignment reference pharmacophore
def readRefPharmacophore(filename: str) -> Pharm.Pharmacophore:
    # create pharmacophore reader instance
    reader = Pharm.PharmacophoreReader(filename)

    # create an instance of the default implementation of the Pharm.Pharmacophore interface
    ph4 = Pharm.BasicPharmacophore()

    try:
        if not reader.read(ph4):  # read reference pharmacophore
            sys.exit("Error: reading reference pharmacophore failed")

    except Exception as e:  # handle exception raised in case of severe read errors
        sys.exit("Error: reading reference pharmacophore failed: " + str(e))

    return ph4


# remove feature orientation informations and set the feature geometry to Pharm.FeatureGeometry.SPHERE
def clearFeatureOrientations(ph4: Pharm.BasicPharmacophore) -> None:
    for ftr in ph4:
        Pharm.clearOrientation(ftr)
        Pharm.setGeometry(ftr, Pharm.FeatureGeometry.SPHERE)


def main() -> None:
    root = "/data/shared/projects/PhectorDB/virtual_screening_ESR1_ant"
    label = "inactives"
    ref_ph4_file = f"{root}/raw/query.pml"
    in_file = f"{root}/raw/{label}.psd"
    out_file = f"{root}/vs/all_{label}_aligned.pt"

    # read the reference pharmacophore
    ref_ph4 = readRefPharmacophore(ref_ph4_file)

    # create reader for input molecules (format specified by file extension)
    #mol_reader = Chem.MoleculeReader(in_file)
    pharm_reader = Pharm.PSDPharmacophoreReader(in_file)

    #Chem.setMultiConfImportParameter(
    #    mol_reader, False
    #)  # treat conformers as individual molecules

    # create an instance of the default implementation of the Chem.Molecule interface
    #mol = Chem.BasicMolecule()
    mol_ph4 = Pharm.BasicPharmacophore()

    # create instance of class implementing the pharmacophore alignment algorithm
    almnt = Pharm.PharmacophoreAlignment(
        True
    )  # True = aligned features have to be within the tolerance spheres of the ref. features

    # clear feature orientation information
    clearFeatureOrientations(ref_ph4)

    almnt.addFeatures(
        ref_ph4, True
    )  # set reference features (True = first set = reference)
    almnt.performExhaustiveSearch(
        False
    )  # set minimum number of top. mapped feature pairs

    # create pharmacophore fit score calculator instance
    almnt_score = Pharm.PharmacophoreFitScore()

    almnt_scores = []

    # read and process molecules one after the other until the end of input has been reached
    try:
        i = 1

        while pharm_reader.read(mol_ph4):
            # compose a simple molecule identifier
            #mol_id = Chem.getName(mol).strip()

            #if mol_id == "":
            #    mol_id = "#" + str(i)  # fallback if name is empty
            #else:
            #    mol_id = "'%s' (#%s)" % (mol_id, str(i))

            # if not args.quiet:
            #print("- Aligning molecule %s..." % mol_id)

            try:
                #mol_ph4 = genPharmacophore(mol)  # generate input molecule pharmacophore

                if mol_ph4.getNumFeatures() == 0:
                    continue

                # if args.pos_only:  # clear feature orientation information
                clearFeatureOrientations(mol_ph4)

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

                print(" -> Found %s alignment solutions" % str(len(almnt_solutions)))

                almnt_solutions = sorted(almnt_solutions, reverse=True)

                if len(almnt_solutions) == 0:
                    almnt_scores.append([0, 0, mol_ph4.getNumFeatures()])

                else:
                    solution = almnt_solutions[0]
                    almnt_scores.append(
                        [
                            int(solution),
                            solution % 1,
                            mol_ph4.getNumFeatures(),
                        ]
                    )

            except Exception as e:
                sys.exit(
                    "Error: pharmacophore alignment of molecule %s failed: %s"
                    #% (mol_id, str(e))
                    % (0, str(e))
                )

            i += 1

    except Exception as e:  # handle exception raised in case of severe read errors
        sys.exit("Error: reading input molecule failed: " + str(e))

    almnt_scores = torch.tensor(almnt_scores)
    torch.save(almnt_scores, out_file)
    print(f"Alignment of {almnt_scores.shape[0]} ph4s")


if __name__ == "__main__":
    tic = time.perf_counter()
    main()
    toc = time.perf_counter()
    print(f"This took {toc - tic:0.4f} seconds")
