import os
import sys
from typing import Optional, Callable

import pandas as pd
import torch
from torch import Tensor
from torch_geometric.data import Data, InMemoryDataset  # , download_url
import CDPL.Pharm as Pharm
from CDPL.Pharm import PSDPharmacophoreReader
import CDPL.Chem as Chem


class PharmacophoreDatasetBase(InMemoryDataset):
    """Abstract Base Class for pharmacophore datasets.

    Extends the InMemoryDataset class from PyTorch Geometric to provide functionality
    for reading pharmacophore data from files and creating Tensor objects from those.
    """

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
    """Implementation of the PharmacophoreDataset class for unlabeled training data.

    Args:
        root (str): Path to the location of the unlabeled pharmacophore dataset.
        transform (Optional[Callable], optional): Parameter of the InMemoryDataset
            class. A function/transform that takes in a `torch_geometric.data.Data`
            object and returns a transformed version. The data object will be
            transformed before every access. Defaults to None.
        pre_transform (Optional[Callable], optional): Parameter of the InMemoryDataset
            class. A function/transform that takes in a `torch_geometric.data.Data`
            object and returns a transformed version. The data object will be
            transformed before being saved to disk. Defaults to None.
        pre_filter (Optional[Callable], optional): Parameter of the InMemoryDataset
            class. A function that takes in a `torch_geometric.data.Data` object and
            returns a boolean value, indicating whether the data object should be
            included in the final dataset. Defaults to None.
    """

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
    """Implementation of the PharmacophoreDataset class for labeled benchmark data.

    Args:
        root (str): Path to the location of the labeled virtual screening dataset.
        path_type (str, optional): VS benchnmark data comes with "active" ligands,
            "inactive" ligands, and a "query". This parameters indicates which data
            shall be retrieved. Defaults to "active".
        transform (Optional[Callable], optional): Parameter of the InMemoryDataset
            class. A function/transform that takes in a `torch_geometric.data.Data`
            object and returns a transformed version. The data object will be
            transformed before every access. Defaults to None.
        pre_transform (Optional[Callable], optional): Parameter of the InMemoryDataset
            class. A function/transform that takes in a `torch_geometric.data.Data`
            object and returns a transformed version. The data object will be
            transformed before being saved to disk. Defaults to None.
        pre_filter (Optional[Callable], optional): Parameter of the InMemoryDataset
            class. A function that takes in a `torch_geometric.data.Data` object and
            returns a boolean value, indicating whether the data object should be
            included in the final dataset. Defaults to None.
    """

    def __init__(
        self,
        root: str,
        path_type: str = "active",
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.path_type = path_type
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

    def get_metadata(self) -> pd.DataFrame:
        if self.path_type == "active":
            path = self.raw_paths[0]
        if self.path_type == "inactive":
            path = self.raw_paths[1]
        if self.path_type == "query":
            path = self.raw_paths[2]
        reader = self._getPharmReaderByFileExt(path)
        ph4 = Pharm.BasicPharmacophore()
        names = []
        features = []
        index = []
        conf_index = []
        num_features = []
        conf = 0
        i = 0
        name = ""

        while reader.read(ph4):
            if ph4.getNumFeatures() == 0:
                continue
            feature_types = Pharm.generateFeatureTypeHistogramString(ph4)
            if name == Pharm.getName(ph4):
                conf += 1
            else:
                conf = 0
                name = Pharm.getName(ph4)
            conf_index.append(conf)
            features.append(feature_types)
            names.append(name)
            index.append(i)
            num_features.append(ph4.getNumFeatures())
            i += 1

        metadata = pd.DataFrame(
            {
                "index": index,
                "name": names,
                "conf_idx": conf_index,
                "features": features,
                "num_features": num_features,
            }
        )

        return metadata
