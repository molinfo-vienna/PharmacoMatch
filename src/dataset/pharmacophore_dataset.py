import os, sys

import torch
from torch_geometric.data import Data, InMemoryDataset  # , download_url
import CDPL.Pharm as Pharm
import CDPL.Chem as Chem

class PharmacophoreDatasetBase(InMemoryDataset):
    _ftr_type_str = { Pharm.FeatureType.UNKNOWN               : 'UNKNOWN',
                    Pharm.FeatureType.HYDROPHOBIC           : 'HYDROPHOBIC',
                    Pharm.FeatureType.AROMATIC              : 'AROMATIC',
                    Pharm.FeatureType.NEGATIVE_IONIZABLE    : 'NEGATIVE_IONIZABLE',
                    Pharm.FeatureType.POSITIVE_IONIZABLE    : 'POSITIVE_IONIZABLE',
                    Pharm.FeatureType.H_BOND_DONOR          : 'H_BOND_DONOR',
                    Pharm.FeatureType.H_BOND_ACCEPTOR       : 'H_BOND_ACCEPTOR',
                    Pharm.FeatureType.HALOGEN_BOND_DONOR    : 'HALOGEN_BOND_DONOR',
                    Pharm.FeatureType.HALOGEN_BOND_ACCEPTOR : 'HALOGEN_BOND_ACCEPTOR',
                    Pharm.FeatureType.EXCLUSION_VOLUME      : 'EXCLUSION_VOLUME' }
    
    _num_node_features = 7
    _num_edge_features = 5

    def get_params(self):
        # input data specific model parameters
        params = dict(
            num_node_features=self._num_node_features,
            num_edge_features=self._num_edge_features,
        )

        return params

    def _extract_pharmacophore_features(self, ph4):
        num_features = ph4.getNumFeatures()
        x = torch.zeros((num_features, self._num_node_features))
        pos = torch.zeros((num_features, 3))

        for i, feature in enumerate(ph4):
            x[i, Pharm.getType(feature)-1] = 1
            pos[i] = torch.tensor(Chem.get3DCoordinates(feature).toArray())

        return x, pos

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
                "Error: unsupported pharmacophore input file format '%s'" % name_and_ext[1]
            )

        # create and return file reader instance
        return ipt_handler.createReader(filename)

class PharmacophoreDataset(PharmacophoreDatasetBase):
    def __init__(
        self, root, transform=None, pre_transform=None, pre_filter=None
    ):
        super().__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0] 
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return ["pretraining_data_large.cdf"]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        pass

    def process(self):
        data_list = self.data_processing(self.raw_paths[0])

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def data_processing(self, path):
        reader = self._getReaderByFileExt(path)
        ph4 = Pharm.BasicPharmacophore()
        data_list = []
        count = 0
        skipped_pharmacophores = 0

        while reader.read(ph4):
            try:
                if ph4.getNumFeatures() > 3: 
                    x, pos = self._extract_pharmacophore_features(ph4)
                    data = Data(x=x, pos=pos)
                    data_list.append(data)
                    count += 1
                else:
                    skipped_pharmacophores += 1
                    # Do not include empty and too small graphs
            
            except Exception as e:
                sys.exit("Error: processing of pharmacophore failed: " + str(e))

        print(f'{skipped_pharmacophores} pharmacophores were rejected.')
        return data_list

    def get_params(self):
        # input data specific model parameters
        params = dict(
            num_node_features=7,
            num_edge_features=5,
        )

        return params


class VirtualScreeningDataset(PharmacophoreDatasetBase):
    def __init__(
        self, root, path_type='active', transform=None, pre_transform=None, pre_filter=None
    ):
        super().__init__(root, transform, pre_transform, pre_filter)
        if path_type == 'active':
            path = self.processed_paths[0] 
        if path_type == 'inactive':
            path = self.processed_paths[1] 
        if path_type == 'query':
            path = self.processed_paths[2] 
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return ["actives.pml", "inactives.pml", "query.pml"]

    @property
    def processed_file_names(self):
        return ["actives.pt", "inactives.pt", "query.pt"]

    def download(self):
        pass

    def process(self):
        for i in range(len(self.raw_paths)):
            data_list = self.data_processing(self.raw_paths[i])

            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]

            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[i])

    def data_processing(self, path):
        reader = self._getReaderByFileExt(path)
        ph4 = Pharm.BasicPharmacophore()
        data_list = []
        name = ''
        mol_id = -1
        count = 0
        skipped_pharmacophores = 0

        while reader.read(ph4):
            try:
                if ph4.getNumFeatures() > 3: 
                    if name != Pharm.getName(ph4):
                        name = Pharm.getName(ph4)
                        mol_id += 1
                    x, pos = self._extract_pharmacophore_features(ph4)
                    data = Data(x=x, pos=pos, mol_id=mol_id)
                    data_list.append(data)
                    count += 1
                else:
                    skipped_pharmacophores += 1
                    # Do not include empty and too small graphs

            except Exception as e:
                sys.exit("Error: processing of pharmacophore failed: " + str(e))

        return data_list
