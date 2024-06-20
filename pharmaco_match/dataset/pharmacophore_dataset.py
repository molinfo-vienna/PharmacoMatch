import os
import sys
from typing import Optional, Callable

import torch
from torch import Tensor
from torch_geometric.data import Data, InMemoryDataset  # , download_url
import CDPL.Pharm as Pharm
from CDPL.Pharm import PSDPharmacophoreReader
import CDPL.Chem as Chem


class PharmacophoreDatasetBase(InMemoryDataset):
    _ftr_type_str = {
        Pharm.FeatureType.UNKNOWN: "UNKNOWN",
        Pharm.FeatureType.HYDROPHOBIC: "HYDROPHOBIC",
        Pharm.FeatureType.AROMATIC: "AROMATIC",
        Pharm.FeatureType.NEGATIVE_IONIZABLE: "NEGATIVE_IONIZABLE",
        Pharm.FeatureType.POSITIVE_IONIZABLE: "POSITIVE_IONIZABLE",
        Pharm.FeatureType.H_BOND_DONOR: "H_BOND_DONOR",
        Pharm.FeatureType.H_BOND_ACCEPTOR: "H_BOND_ACCEPTOR",
        Pharm.FeatureType.HALOGEN_BOND_DONOR: "HALOGEN_BOND_DONOR",
        Pharm.FeatureType.HALOGEN_BOND_ACCEPTOR: "HALOGEN_BOND_ACCEPTOR",
        Pharm.FeatureType.EXCLUSION_VOLUME: "EXCLUSION_VOLUME",
    }

    _num_node_features = 7

    def _extract_pharmacophore_features(
        self, ph4: Pharm.BasicPharmacophore
    ) -> tuple[Tensor, Tensor]:
        num_ph4_features = ph4.getNumFeatures()
        x = torch.zeros((num_ph4_features, self._num_node_features))
        pos = torch.zeros((num_ph4_features, 3))

        for i, feature in enumerate(ph4):
            x[i, Pharm.getType(feature) - 1] = 1
            pos[i] = torch.tensor(Chem.get3DCoordinates(feature).toArray())

        return x, pos, num_ph4_features

    def _getMolReaderByFileExt(self, filename: str) -> Chem.MoleculeReader:
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
            sys.exit(
                "Error: unsupported molecule input file format '%s'" % name_and_ext[1]
            )

        # create and return file reader instance
        return ipt_handler.createReader(filename)

    def _getPharmReaderByFileExt(self, filename: str) -> Pharm.PharmacophoreReader:
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


class PharmacophoreDataset(PharmacophoreDatasetBase):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self) -> list[str]:
        return ["chembl_data.cdf"]

    @property
    def processed_file_names(self) -> list[str]:
        return ["chembl_data.pt"]

    def download(self) -> None:
        pass

    def process(self) -> None:
        data_list = self.data_processing(self.raw_paths[0])

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def data_processing(self, path: str) -> list[Data]:
        reader = self._getPharmReaderByFileExt(path)
        ph4 = Pharm.BasicPharmacophore()
        data_list = []
        count = 0
        skipped_pharmacophores = 0

        while reader.read(ph4):
            try:
                # Do not include empty and too small graphs
                if ph4.getNumFeatures() > 3:
                    x, pos, num_ph4_features = self._extract_pharmacophore_features(ph4)
                    data = Data(x=x, pos=pos, num_ph4_features=num_ph4_features)
                    data_list.append(data)
                    count += 1
                else:
                    skipped_pharmacophores += 1

            except Exception as e:
                sys.exit("Error: processing of pharmacophore failed: " + str(e))

        print(f"{skipped_pharmacophores} pharmacophores were rejected.")

        return data_list


class VirtualScreeningDataset(PharmacophoreDatasetBase):
    def __init__(
        self,
        root: str,
        path_type: str = "active",
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        super().__init__(root, transform, pre_transform, pre_filter)
        if path_type == "active":
            path = self.processed_paths[0]
        if path_type == "inactive":
            path = self.processed_paths[1]
        if path_type == "query":
            path = self.processed_paths[2]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self) -> list[str]:
        return ["actives.psd", "inactives.psd", "query.pml"]

    @property
    def processed_file_names(self) -> list[str]:
        return ["actives.pt", "inactives.pt", "query.pt"]

    def download(self) -> None:
        pass

    def process(self) -> None:
        for i in range(len(self.raw_paths)):
            data_list = self.data_processing(self.raw_paths[i])

            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]

            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[i])

    def data_processing(self, path: str) -> list[str]:
        pharm_reader = self._getPharmReaderByFileExt(path)

        if type(pharm_reader) is PSDPharmacophoreReader:
            db_accessor = Pharm.PSDScreeningDBAccessor(path)
            ph4 = Pharm.BasicPharmacophore()
            num_molecules = db_accessor.getNumMolecules()
            data_list = []
            skipped_pharmacophores = 0

            for i in range(num_molecules):
                try:
                    num_pharmacophores = db_accessor.getNumPharmacophores(i)
                    for j in range(num_pharmacophores):
                        db_accessor.getPharmacophore(i, j, ph4)
                        # Do not include empty graphs
                        if ph4.getNumFeatures() > 0:
                            x, pos, num_ph4_features = (
                                self._extract_pharmacophore_features(ph4)
                            )
                            data = Data(
                                x=x,
                                pos=pos,
                                mol_id=i,
                                num_ph4_features=num_ph4_features,
                            )
                            data_list.append(data)
                        else:
                            skipped_pharmacophores += 1

                except Exception as e:
                    sys.exit("Error: processing of pharmacophore failed: " + str(e))

            return data_list

        else:
            ph4 = Pharm.BasicPharmacophore()
            data_list = []
            name = ""
            mol_id = -1
            skipped_pharmacophores = 0

            while pharm_reader.read(ph4):
                try:
                    # Do not include empty graphs
                    if ph4.getNumFeatures() > 0:
                        if name != Pharm.getName(ph4):
                            name = Pharm.getName(ph4)
                            mol_id += 1
                        x, pos, num_ph4_features = self._extract_pharmacophore_features(
                            ph4
                        )
                        data = Data(
                            x=x,
                            pos=pos,
                            mol_id=mol_id,
                            num_ph4_features=num_ph4_features,
                        )
                        data_list.append(data)
                    else:
                        skipped_pharmacophores += 1

                except Exception as e:
                    sys.exit("Error: processing of pharmacophore failed: " + str(e))

            return data_list
