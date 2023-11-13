import os, sys

import torch
from torch_geometric.data import Data, InMemoryDataset  # , download_url
import CDPL.Pharm as Pharm
import CDPL.Chem as Chem


class PharmacophoreDataset(InMemoryDataset):
    def __init__(
        self, root, small_set=False, transform=None, pre_transform=None, pre_filter=None
    ):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.small_set = small_set
        path = self.processed_paths[0] 
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return ["pubchem-10m-clean.cdf"]

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
        reader = getReaderByFileExt(path)
        ph4 = Pharm.BasicPharmacophore()
        data_list = []
        count = 0

        while reader.read(ph4):
            try:
                # get number of features
                num_features = ph4.getNumFeatures()
                if num_features <= 0: break # Do not include empty graphs

                x = torch.zeros((num_features, 9))
                pos = torch.zeros((num_features, 3))

                for i, feature in enumerate(ph4):
                    x[i, Pharm.getType(feature)] = 1
                    pos[i] = torch.tensor(Chem.get3DCoordinates(feature).toArray())

                data = Data(x=x, pos=pos)
                data_list.append(data)
                count += 1
                if self.small_set and count >= 1e5: break

            except Exception as e:
                sys.exit("Error: processing of pharmacophore failed: " + str(e))

        return data_list

    def get_params(self):
        # input data specific model parameters
        params = dict(
            num_node_features=self.num_node_features,
            num_edge_features=self.num_edge_features,
        )

        return params


class VirtualScreeningDataset(InMemoryDataset):
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
        reader = getReaderByFileExt(path)
        ph4 = Pharm.BasicPharmacophore()
        data_list = []
        name = ''
        mol_id = -1

        while reader.read(ph4):
            try:
                # get number of features
                num_features = ph4.getNumFeatures()
                if num_features <= 0: break # Do not include empty graphs

                x = torch.zeros((num_features, 9))
                pos = torch.zeros((num_features, 3))

                for i, feature in enumerate(ph4):
                    x[i, Pharm.getType(feature)] = 1
                    pos[i] = torch.tensor(Chem.get3DCoordinates(feature).toArray())

                if name != Pharm.getName(ph4):
                    name = Pharm.getName(ph4)
                    mol_id += 1

                data = Data(x=x, pos=pos, mol_id = mol_id)
                data_list.append(data)

            except Exception as e:
                sys.exit("Error: processing of pharmacophore failed: " + str(e))

        return data_list

    def get_params(self):
        # input data specific model parameters
        params = dict(
            num_node_features=self.num_node_features,
            num_edge_features=self.num_edge_features,
        )

        return params


def getReaderByFileExt(filename: str) -> Pharm.PharmacophoreReader:
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


