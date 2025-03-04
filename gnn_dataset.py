import numpy as np
import pandas as pd
import os

from sklearn.preprocessing import LabelEncoder
import sklearn
from sklearn import preprocessing

import torch
import json
import joblib

from torch_geometric.data.data import Data
from torch.utils.data import Dataset
from torch_geometric.data import Batch, InMemoryDataset

from gnn_graph import graph_from_smiles


def minmax_to_dict(Scaler):
    minmax_dict = {}
    minmax_dict["min_"] = list(Scaler.min_)
    minmax_dict["scale_"] = list(Scaler.scale_)
    minmax_dict["data_min_"] = list(Scaler.data_min_)
    minmax_dict["data_max_"] = list(Scaler.data_max_)
    minmax_dict["data_range_"] = list(Scaler.data_range_)
    minmax_dict["n_features_in_"] = Scaler.n_features_in_
    minmax_dict["n_samples_seen_"] = Scaler.n_samples_seen_
    # minmax_dict["feature_names_in_"] = Scaler.feature_names_in
    return minmax_dict

def minmax_to_json(Scaler, filename):
    minmax_dict = minmax_to_dict(Scaler)
    with open(filename, "w") as outfile:
        json.dump(minmax_dict, outfile,
                  indent=4, sort_keys=False)
    return


def minmax_from_dict(minmax_dict):
    Scaler = sklearn.preprocessing.MinMaxScaler()
    a = np.atleast_2d(minmax_dict["data_min_"])
    b = np.atleast_2d(minmax_dict["data_max_"])
    c = np.concatenate((a, b))
    return Scaler.fit(c)


def minmax_from_json(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    return minmax_from_dict(data)


class GraphDataset(InMemoryDataset):
    def __init__(self, root):
        super().__init__(root)
        self.load(self.processed_paths[0])
        self.scaler = joblib.load(self.processed_paths[1])
        self.info_df = pd.read_csv(self.processed_paths[2])
        self.identifier = "isomeric_smiles"
        return

    @property
    def raw_file_names(self):
        return ['ff_parameters.csv', 'raw_data.csv']

    @property
    def processed_file_names(self):
        # dummy = ['data.pt','scaler.gz']
        # return [ os.path.join(self.root,"processed",d) for d in dummy ]
        return ['graph_data.pt', 'graph_scaler.gz', 'graph_list.csv']

    def download(self):
        # Download to `self.raw_dir`.
        # download_url(url, self.raw_dir)
        print("no download available")
        return

    def get_graph_id(self, x):
        try:
            return self.info_df[self.info_df["identifier"] == x["identifier"]].iloc[0].id
        except:
            print("WARNING: no id found: ")
            print(x["identifier"], x["iupac_name"])
            return -1

    def process(self):
        # Read data into huge `Data` list.
        # print("self.root",self.root)
        # print("self.pre_transform",self.pre_transform)
        # print("self.raw_file_names",self.raw_file_names)
        self.identifier = "isomeric_smiles"
        if not os.path.isfile(self.processed_paths[0]):
            ff_keys = ["epsilon", "sigma", "charge", "neighbors charge"]

            ff_data = pd.read_csv(self.raw_paths[0])
            data = pd.read_csv(self.raw_paths[1])

            self.scaler = joblib.load(self.processed_paths[1])

            data["identifier"] = data[self.identifier]
            
            data_list = []
            ids = []
            smiles_processed = []
            identifier = []
            usmiles = np.unique( data["isomeric_smiles"] )
            ii = 0
            for _, smiles in enumerate(usmiles):
                #try:
                row = data[ data["isomeric_smiles"] == smiles].iloc[0]
                print(smiles)
                atom_embeddings, bond_list = graph_from_smiles(smiles, ff_data,
                                                              ff_keys=ff_keys
                                                              )
                atom_embeddings = self.scaler.transform(atom_embeddings)
                """
                build graph
                """
                atom_embeddings = torch.tensor(atom_embeddings)
                bond_list = torch.tensor(bond_list.T)
                graph_len = atom_embeddings.shape[0]
                # yy = torch.tensor(atom_weights)
                graph = Data(x=atom_embeddings.float(), edge_index=bond_list, 
                             graph_len=graph_len, smiles=smiles, index=ii,
                             ones=torch.ones(graph_len)
                            )
                data_list.append(graph)
                ids.append(ii)
                smiles_processed.append(smiles)
                identifier.append(row["identifier"])
                ii += 1
                #except:
                #    print()
                #    print(smiles)
                #    print("ERROR")
                #    print()
            ddict = {"identifier": identifier, "id": ids, "smiles": smiles_processed}
            self.info_df = pd.DataFrame(ddict)
            self.info_df.to_csv(self.processed_paths[2])

        # else:
        #    self.scaler = joblib.load(self.processed_paths[1])

        # if self.pre_filter is not None:
        #    data_list = [data for data in data_list if self.pre_filter(data)]

        # if self.pre_transform is not None:
        #    data_list = [self.pre_transform(data) for data in data_list]

            self.save(data_list, self.processed_paths[0])
        # For PyG<2.4:
        # torch.save(self.collate(data_list), self.processed_paths[0])
        return


class dataset(Dataset):
    """
    - standard dataset for final training
    - returns random batches
    """
    # Constructor with defult values
    def __init__(self, root, data_csv, log_transform=True,
                 x_features=["temperature", "resd_entropy"],
                 y_features=["log_value"],
                 keep_features=["identifier","iupac_name", "cas","family","graph_id"],
                 keepXY=False, scale_x_features_with=[],
                 ):
        self.root = root
        self.x_features = x_features
        self.y_features = y_features
        self.keep_features = keep_features

        self.graph_data = GraphDataset(self.root)
        self.identifier = self.graph_data.identifier
        
        data_csv_processed = os.path.join(self.root, "processed",data_csv)
        if os.path.isfile(data_csv_processed):
            self.data_csv = data_csv_processed
            data = pd.read_csv(self.data_csv)
        else:
            self.data_csv = os.path.join(self.root, "raw", data_csv)
            data = pd.read_csv(self.data_csv)
            data["identifier"] = data[self.identifier]
            data['graph_id'] = np.array(data.apply(self.graph_data.get_graph_id, axis=1))

            data = data[data["graph_id"] >= 0]
            if "resd_entropy" in data.keys():
                data["resd_entropy"] = np.abs(data["resd_entropy"])
            if "pressure" in data.keys():
                data = data[data["pressure"] > 0]
                data["log_pressure"] = np.log(data["pressure"])
            if log_transform:
                data["log_value"] = np.log(data["value"])

            data.to_csv(data_csv_processed)

        self.graph_indexes = np.array(data["graph_id"])

        X = np.array(data[x_features])
        Y = np.array(data[y_features])
        if keepXY:
            self.X = X
            self.Y = Y
        else:
            self.X = None
            self.Y = None         

        # norm data here
        self.scalerX_path = os.path.join(self.root, "processed", "data_scalerX.gz")
        self.scalerY_path = os.path.join(self.root, "processed", "data_scalerY.gz")
        if not os.path.isfile(self.scalerX_path) or not os.path.isfile(self.scalerY_path):
            if len(scale_x_features_with):
                dummy = np.array(data[scale_x_features_with])
                self.scalerX = preprocessing.MinMaxScaler().fit(dummy)
                # self.scalerX = preprocessing.StandardScaler().fit(dummy)
            else:
                self.scalerX = preprocessing.MinMaxScaler().fit(X)
                #  self.scalerX = preprocessing.StandardScaler().fit(X)
            self.scalerY = preprocessing.MinMaxScaler().fit(Y)
            # self.scalerY = preprocessing.StandardScaler().fit(Y)
            joblib.dump(self.scalerX, self.scalerX_path)
            joblib.dump(self.scalerY, self.scalerY_path)
        else:
            self.scalerX = joblib.load(self.scalerX_path)
            self.scalerY = joblib.load(self.scalerY_path)

        self.X_scaled = torch.Tensor(self.scalerX.transform(X))
        self.Y_scaled = torch.Tensor(self.scalerY.transform(Y))

        self.keep_path = os.path.join(self.root, "processed", "keep.csv")
        if not os.path.isfile(self.keep_path):
            self.keep = data[keep_features]
            self.keep.to_csv(self.keep_path)
        else:
            self.keep = pd.read_csv(self.keep_path)

        self.encoded_species = LabelEncoder().fit_transform(data["identifier"])
        self.encoded_families = LabelEncoder().fit_transform(data["family"])

        self.n_species = np.unique(self.encoded_species).shape[0]
        self.n_families = np.unique(self.encoded_families).shape[0]

        self.len = self.Y_scaled.shape[0]

        self.species_indexes = []
        for n in np.unique(self.encoded_species):
            p = np.where(self.encoded_species == n)
            self.species_indexes.append(np.squeeze(p))

        self.families_indexes = []
        for n in np.unique(self.encoded_families):
            p = np.where(self.encoded_families == n)
            self.families_indexes.append(np.squeeze(p))

        del data
        return

    # Getter
    def __getitem__(self, index):
        batch = self.graph_indexes[index]
        sample = self.X_scaled[index].float(), self.Y_scaled[index].float(), batch
        return sample

    # Get Length
    def __len__(self):
        return self.len

    def build_batch(self, index):
        batch = Batch.from_data_list(
                self.graph_data[index]
            )
        return batch

    def get_keep(self, index):
        return self.keep.iloc[index]

    def get_data(self):
        return self.X_scaled.float(), self.Y_scaled.float()

    # Getter
    def get_species(self, index):
        p = self.species_indexes[index]
        batch = self.graph_indexes[p]
        sample = self.X_scaled[p].float(), self.Y_scaled[p].float(), batch
        return sample

    def get_species_keep(self, index):
        return self.keep.iloc[self.species_indexes[index]]
