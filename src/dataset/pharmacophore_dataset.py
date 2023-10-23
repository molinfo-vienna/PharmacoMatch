import os, sys

import torch
from torch_geometric.data import Data, InMemoryDataset  # , download_url
import CDPL.Pharm as Pharm
import CDPL.Chem as Chem


class PharmacophoreDataset(InMemoryDataset):
    def __init__(
        self, root, train=True, transform=None, pre_transform=None, pre_filter=None
    ):
        super().__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
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

        while reader.read(ph4):
            try:
                # get number of features
                num_features = ph4.getNumFeatures()
                x = torch.zeros((num_features, 9))
                pos = torch.zeros((num_features, 3))
                for i, feature in enumerate(ph4):
                    x[i, Pharm.getType(feature)] = 1
                    pos[i] = torch.tensor(Chem.get3DCoordinates(feature).toArray())
                # edge_index = knn_graph(pos, k=100)
                data = Data(x=x, pos=pos)
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
