import argparse
import math
import os
import os.path as osp
import random
from copy import deepcopy

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures, Draw, MolFromSmiles
from torch_geometric import data as DATA
from torch_geometric.data import InMemoryDataset
import torch_geometric.transforms as T
from tqdm import tqdm
import pdb
from model import ColdDTA

fdef_name = osp.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)


VOCAB_PROTEIN = { "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, 
				"F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12, 
				"O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18, 
				"U": 19, "T": 20, "W": 21, 
				"V": 22, "Y": 23, "X": 24, 
				"Z": 25 }

def seqs2int(target):

    return [VOCAB_PROTEIN[s] for s in target.upper()] 


def atom_features(atom):
    encoding = one_of_k_encoding_unk(atom.GetSymbol(), ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown'])
    encoding += one_of_k_encoding(atom.GetDegree(), [0,1,2,3,4,5,6,7,8,9,10]) + one_of_k_encoding_unk(atom.GetTotalNumHs(), [0,1,2,3,4,5,6,7,8,9,10]) 
    encoding += one_of_k_encoding_unk(atom.GetImplicitValence(), [0,1,2,3,4,5,6,7,8,9,10]) 
    encoding += one_of_k_encoding_unk(atom.GetHybridization(), [
                      Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                      Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                      Chem.rdchem.HybridizationType.SP3D2, 'other']) 
    encoding += [atom.GetIsAromatic()]

    try:
        encoding += one_of_k_encoding_unk(
                    atom.GetProp('_CIPCode'),
                    ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
    except:
        encoding += [0, 0] + [atom.HasProp('_ChiralityPossible')]
    
    return np.array(encoding)


def remove_subgraph(Graph, center, percent):
    assert percent <= 1
    G = Graph.copy()
    num = int(np.floor(len(G.nodes())*percent))
    removed = []
    temp = [center]
    while len(removed) < num and temp:
        neighbors = []

        try:
            for n in temp:
                neighbors.extend([i for i in G.neighbors(n) if i not in temp])      
        except Exception as e:
            print(e)
            return None, None

        for n in temp:
            if len(removed) < num:
                G.remove_node(n)
                removed.append(n)
            else:
                break

        temp = list(set(neighbors))

    return G, removed

# data augmentation
def mol_to_graph(mol, times=2):
    start_list = random.sample(list(range(mol.GetNumAtoms())), times)

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])

    molGraph = nx.Graph(edges)
    percent = 0.2
    removed_list = []
    for i in range(1, times + 1):
        G, removed = remove_subgraph(molGraph, start_list[i - 1], percent)
        removed_list.append(removed)
    # pdb.set_trace()

    for removed_i in removed_list:
        if not removed_i:
            return None, None, None

    features_list = []
    for i in range(times):
        features_list.append([])

    for index, atom in enumerate(mol.GetAtoms()):
        for i, removed_i in enumerate(removed_list):
            if index not in removed_i:
                feature = atom_features(atom)
                features_list[i].append(feature/np.sum(feature))

    edges_list = []
    for i in range(times):
        edges_list.append([])

    g = nx.DiGraph()
    for i in range(mol.GetNumAtoms()):
        for j in range(mol.GetNumAtoms()):
            e_ij = mol.GetBondBetweenAtoms(i, j)
            if e_ij is not None:
                g.add_edge(i, j,
                            b_type=e_ij.GetBondType(),
                            IsConjugated=int(e_ij.GetIsConjugated()),
                            )
    
    edge_attr_list = []
    for i in range(times):
        edge_attr_list.append([])
    for i, removed_i in enumerate(removed_list):
        e = {}
        for n1, n2, d in g.edges(data=True):
        
            if n1 not in removed_i and n2 not in removed_i:
                start_i = n1 - sum(num < n1 for num in removed_i)
                e_i = n2 - sum(num < n2 for num in removed_i)
                edges_list[i].append([start_i, e_i])
                e_t = [int(d['b_type'] == x)
                        for x in (Chem.rdchem.BondType.SINGLE, \
                                    Chem.rdchem.BondType.DOUBLE, \
                                    Chem.rdchem.BondType.TRIPLE, \
                                    Chem.rdchem.BondType.AROMATIC)]
                e_t.append(int(d['IsConjugated'] == False))
                e_t.append(int(d['IsConjugated'] == True))
                e[(n1, n2)] = e_t
        edge_attr = list(e.values())
        edge_attr_list[i] = edge_attr

    if len(e) == 0:
        return features_list, torch.LongTensor([[0], [0]]), torch.FloatTensor([[0, 0, 0, 0, 0, 0]])

    edge_index_list = []
    for i in range(times):
        edge_index_list.append([])
    for i, edges_i in enumerate(edges_list):
        g_i = nx.Graph(edges_i).to_directed()
        for e1, e2 in g_i.edges():
            edge_index_list[i].append([e1, e2])

    return features_list, edge_index_list, edge_attr_list


def mol_to_graph_without_rm(mol):
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature/np.sum(feature))

    g = nx.DiGraph()
    for i in range(mol.GetNumAtoms()):
        for j in range(mol.GetNumAtoms()):
            e_ij = mol.GetBondBetweenAtoms(i, j)
            if e_ij is not None:
                g.add_edge(i, j,
                            b_type=e_ij.GetBondType(),
                            IsConjugated=int(e_ij.GetIsConjugated()),
                            )
    e = {}
    for n1, n2, d in g.edges(data=True):
        e_t = [int(d['b_type'] == x)
                for x in (Chem.rdchem.BondType.SINGLE, \
                            Chem.rdchem.BondType.DOUBLE, \
                            Chem.rdchem.BondType.TRIPLE, \
                            Chem.rdchem.BondType.AROMATIC)]
        e_t.append(int(d['IsConjugated'] == False))
        e_t.append(int(d['IsConjugated'] == True))
        e[(n1, n2)] = e_t

    if len(e) == 0:
        return features, torch.LongTensor([[0], [0]]), torch.FloatTensor([[0, 0, 0, 0, 0, 0]])

    edge_index = torch.LongTensor(list(e.keys())).transpose(0, 1)
    edge_attr = torch.FloatTensor(list(e.values()))
    # pdb.set_trace()
    return features, edge_index, edge_attr


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

class GNNDataset(InMemoryDataset):
    def __init__(self, root, index=0, types='train', transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)    

    @property
    def raw_file_names(self):
        return ['davis_fold_0.csv', 'davis_fold_1.csv', 'davis_fold_2.csv', 'davis_fold_3.csv', 'davis_fold_4.csv']

    @property
    def processed_file_names(self):
        return ['processed_data_fold_0.pt', 'processed_data_fold_1.pt', 'processed_data_fold_2.pt', 'processed_data_fold_3.pt','processed_data_fold_4.pt', 
        'processed_test_data_fold_0.pt', 'processed_test_data_fold_1.pt', 'processed_test_data_fold_2.pt', 'processed_test_data_fold_3.pt','processed_test_data_fold_4.pt']

    def download(self):
        pass


    def process(self):
        # pdb.set_trace()
        fold_0_list = self.process_train_data(self.raw_paths[0])
        fold_1_list = self.process_train_data(self.raw_paths[1])
        fold_2_list = self.process_train_data(self.raw_paths[2])
        fold_3_list = self.process_train_data(self.raw_paths[3])
        fold_4_list = self.process_train_data(self.raw_paths[4])

        fold_0_test_list = self.process_data(self.raw_paths[0])
        fold_1_test_list = self.process_data(self.raw_paths[1])
        fold_2_test_list = self.process_data(self.raw_paths[2])
        fold_3_test_list = self.process_data(self.raw_paths[3])
        fold_4_test_list = self.process_data(self.raw_paths[4])

        print('Graph construction done. Saving to file.')

        # save preprocessed train data:
        data, slices = self.collate(fold_0_list)
        torch.save((data, slices), self.processed_paths[0])
        data, slices = self.collate(fold_1_list)
        torch.save((data, slices), self.processed_paths[1])
        data, slices = self.collate(fold_2_list)
        torch.save((data, slices), self.processed_paths[2])
        data, slices = self.collate(fold_3_list)
        torch.save((data, slices), self.processed_paths[3])
        data, slices = self.collate(fold_4_list)
        torch.save((data, slices), self.processed_paths[4])

        # save preprocessed test data:
        data, slices = self.collate(fold_0_test_list)
        torch.save((data, slices), self.processed_paths[5])
        data, slices = self.collate(fold_1_test_list)
        torch.save((data, slices), self.processed_paths[6])
        data, slices = self.collate(fold_2_test_list)
        torch.save((data, slices), self.processed_paths[7])
        data, slices = self.collate(fold_3_test_list)
        torch.save((data, slices), self.processed_paths[8])
        data, slices = self.collate(fold_4_test_list)
        torch.save((data, slices), self.processed_paths[9])

    def process_data(self, data_path):
        pdb.set_trace()
        df = pd.read_csv(data_path)
        data_list = []
        delete_list = []
        for i, row in df.iterrows():
            smi = row['SMILES']
            sequence = row['target_sequence'].upper()
            affinity = row['affinity']
            mol = Chem.MolFromSmiles(smi)
            if mol == None:
                print("Unable to process: ", smi)
                continue
            x, edge_index, edge_attr = mol_to_graph_without_rm(mol)

            target = seqs2int(sequence)
            target_len = 1200
            if len(target) < target_len:
                target = np.pad(target, (0, target_len - len(target)))
            else:
                target = target[:target_len]

            data = DATA.Data(
                x=torch.FloatTensor(x),
                smi=smi,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.FloatTensor([affinity]),
                target=torch.LongTensor([target])
            )
            data_list.append(data)
            
        if len(delete_list) > 0:
            df = df.drop(delete_list, axis=0, inplace=False)
            df.to_csv(data_path, index=False)
        print('mol_to_graph_without_rm')
        return data_list

    def process_train_data(self, data_path):
        pdb.set_trace()
        df = pd.read_csv(data_path)
        data_list = []
        delete_list = []
        for i, row in df.iterrows():
            smi = row['SMILES']
            sequence = row['target_sequence'].upper()
            affinity = row['affinity']
            mol = Chem.MolFromSmiles(smi)
            if mol == None:
                print("Unable to process: ", smi)
                continue

            x_list, edge_index_list, edge_attr_list= mol_to_graph(mol, times = 2)

            target = seqs2int(sequence)
            target_len = 1200
            if len(target) < target_len:
                target = np.pad(target, (0, target_len - len(target)))
            else:
                target = target[:target_len]
            
            if x_list and edge_index_list:
                for x, edge_index, edge_attr in zip(x_list, edge_index_list, edge_attr_list):
                    try:
                        data = DATA.Data(
                            x=torch.FloatTensor(x),
                            smi=smi,
                            edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                            edge_attr=torch.FloatTensor(edge_attr),
                            y=torch.FloatTensor([affinity]),
                            target=torch.LongTensor([target])
                        )
                        data_list.append(data)
                    except Exception as e:
                        print(e)
                        continue

        print('process train data')
        if len(delete_list) > 0:
            df = df.drop(delete_list, axis=0, inplace=False)
            df.to_csv(data_path, index=False)

        return data_list

class GNNDataset2(InMemoryDataset):
    def __init__(self, root, index=0, types='train', transform=None, pre_transform=None, pre_filter=None):
        self.dataset = root.split("/")[1]
        self.root = root
        self.process()

    @property
    def raw_file_names(self):
        return [f'data/{self.dataset}/{self.dataset}_train.csv', 
                f'data/{self.dataset}/{self.dataset}_test.csv']

    @property
    def processed_file_names(self):
        return [f'{self.dataset}_train.pt', f'{self.dataset}_test.pt']

    def download(self):
        pass


    def process(self):
        train_list = self.process_train_data(self.raw_file_names[0])
        test_list = self.process_test_data(self.raw_file_names[1])
        print('Graph construction done. Saving to file.')

        # save preprocessed train data:
        data, slices = self.collate(train_list)
        torch.save((data, slices), f'{self.root}/processed/{self.processed_file_names[0]}')

        # save preprocessed test data:
        data, slices = self.collate(test_list)
        torch.save((data, slices), f'{self.root}/processed/{self.processed_file_names[1]}')

    def process_test_data(self, data_path):
        print(f"Processing {data_path}")
        # pdb.set_trace()
        df = pd.read_csv(data_path)
        data_list = []
        delete_list = []
        for i, row in tqdm(df.iterrows()):
            smi = row['compound_iso_smiles']
            sequence = row['target_sequence'].upper()
            affinity = row['affinity']
            protein_name = row['protein_name']
            mol = Chem.MolFromSmiles(smi)
            if mol == None:
                print("Unable to process: ", smi)
                continue
            x, edge_index, edge_attr = mol_to_graph_without_rm(mol)

            target = seqs2int(sequence)
            target_len = 1200
            if len(target) < target_len:
                target = np.pad(target, (0, target_len - len(target)))
            else:
                target = target[:target_len]

            data = DATA.Data(
                x=torch.FloatTensor(x),
                smi=smi,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.FloatTensor([affinity]),
                target=torch.LongTensor([target]),
                protein_name=protein_name
            )
            data_list.append(data)

        # if len(delete_list) > 0:
            # df = df.drop(delete_list, axis=0, inplace=False)
            # df.to_csv(data_path, index=False)
        print('mol_to_graph_without_rm')
        return data_list

    def process_train_data(self, data_path):
        print(f"Processing {data_path}")
        # pdb.set_trace()
        df = pd.read_csv(data_path)
        data_list = []
        delete_list = []
        for i, row in tqdm(df.iterrows()):
            smi = row['compound_iso_smiles']
            sequence = row['target_sequence'].upper()
            affinity = row['affinity']
            protein_name = row['protein_name']
            mol = Chem.MolFromSmiles(smi)
            if mol == None:
                print("Unable to process: ", smi)
                continue

            x_list, edge_index_list, edge_attr_list = mol_to_graph(mol, times = 2)

            target = seqs2int(sequence)
            target_len = 1200
            if len(target) < target_len:
                target = np.pad(target, (0, target_len - len(target)))
            else:
                target = target[:target_len]
            
            if x_list and edge_index_list:
                for x, edge_index, edge_attr in zip(x_list, edge_index_list, edge_attr_list):
                    try:
                        data = DATA.Data(
                            x=torch.FloatTensor(x),
                            smi=smi,
                            edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                            edge_attr=torch.FloatTensor(edge_attr),
                            y=torch.FloatTensor([affinity]),
                            target=torch.LongTensor([target]),
                            protein_name=protein_name
                        )
                        data_list.append(data)
                        # pdb.set_trace()
                    except Exception as e:
                        print(e)
                        continue

        print('process train data')
        # if len(delete_list) > 0:
            # df = df.drop(delete_list, axis=0, inplace=False)
            # df.to_csv(data_path, index=False)

        return data_list
    
class GNNDataset3(InMemoryDataset):
    def __init__(self, root, index=0, types='train', transform=None, pre_transform=None, pre_filter=None):
        # self.dataset = 'BindingDB_Kd_4S'
        self.dataset = root.split("/")[1]
        self.root = root
        self.process()

    @property
    def raw_file_names(self):
        return [
            f'data/{self.dataset}/train300.csv', 
            f'data/{self.dataset}/test_S1.csv',
            f'data/{self.dataset}/test_S2.csv',
            f'data/{self.dataset}/test_S3.csv',
            f'data/{self.dataset}/test_S4.csv'
        ]

    @property
    def processed_file_names(self):
        return [
            f'{self.dataset}_train.pt', 
            f'{self.dataset}_test_S1.pt',
            f'{self.dataset}_test_S2.pt',
            f'{self.dataset}_test_S3.pt',
            f'{self.dataset}_test_S4.pt'
        ]

    def download(self):
        pass


    def process(self):
        train_list = self.process_train_data(self.raw_file_names[0])
        test_S1_list = self.process_test_data(self.raw_file_names[1])
        test_S2_list = self.process_test_data(self.raw_file_names[2])
        test_S3_list = self.process_test_data(self.raw_file_names[3])
        test_S4_list = self.process_test_data(self.raw_file_names[4])
        print('Graph construction done. Saving to file.')

        # save preprocessed train data:
        data, slices = self.collate(train_list)
        torch.save((data, slices), f'{self.root}/processed/{self.processed_file_names[0]}')

        # save preprocessed test data:
        data, slices = self.collate(test_S1_list)
        torch.save((data, slices), f'{self.root}/processed/{self.processed_file_names[1]}')
        data, slices = self.collate(test_S2_list)
        torch.save((data, slices), f'{self.root}/processed/{self.processed_file_names[2]}')
        data, slices = self.collate(test_S3_list)
        torch.save((data, slices), f'{self.root}/processed/{self.processed_file_names[3]}')
        data, slices = self.collate(test_S4_list)
        torch.save((data, slices), f'{self.root}/processed/{self.processed_file_names[4]}')

    def process_test_data(self, data_path):
        # pdb.set_trace()
        df = pd.read_csv(data_path)
        data_list = []
        delete_list = []
        for i, row in tqdm(df.iterrows()):
            smi = row['compound_iso_smiles']
            sequence = row['target_sequence'].upper()
            affinity = row['affinity']
            protein_name = row['protein_name']
            mol = Chem.MolFromSmiles(smi)
            if mol == None:
                print("Unable to process: ", smi)
                continue
            x, edge_index, edge_attr = mol_to_graph_without_rm(mol)

            target = seqs2int(sequence)
            target_len = 1200
            if len(target) < target_len:
                target = np.pad(target, (0, target_len - len(target)))
            else:
                target = target[:target_len]

            data = DATA.Data(
                x=torch.FloatTensor(x),
                smi=smi,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.FloatTensor([affinity]),
                target=torch.LongTensor([target]),
                protein_name=protein_name
            )
            data_list.append(data)

        # if len(delete_list) > 0:
            # df = df.drop(delete_list, axis=0, inplace=False)
            # df.to_csv(data_path, index=False)
        print('mol_to_graph_without_rm')
        return data_list

    def process_train_data(self, data_path):
        # pdb.set_trace()
        df = pd.read_csv(data_path)
        data_list = []
        delete_list = []
        for i, row in tqdm(df.iterrows()):
            smi = row['compound_iso_smiles']
            sequence = row['target_sequence'].upper()
            affinity = row['affinity']
            protein_name = row['protein_name']
            mol = Chem.MolFromSmiles(smi)
            if mol == None:
                print("Unable to process: ", smi)
                continue

            x_list, edge_index_list, edge_attr_list = mol_to_graph(mol, times = 2)

            target = seqs2int(sequence)
            target_len = 1200
            if len(target) < target_len:
                target = np.pad(target, (0, target_len - len(target)))
            else:
                target = target[:target_len]
            
            if x_list and edge_index_list:
                for x, edge_index, edge_attr in zip(x_list, edge_index_list, edge_attr_list):
                    try:
                        data = DATA.Data(
                            x=torch.FloatTensor(x),
                            smi=smi,
                            edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                            edge_attr=torch.FloatTensor(edge_attr),
                            y=torch.FloatTensor([affinity]),
                            target=torch.LongTensor([target]),
                            protein_name=protein_name
                        )
                        data_list.append(data)
                        pdb.set_trace()
                    except Exception as e:
                        print(e)
                        continue

        print('process train data')
        # if len(delete_list) > 0:
            # df = df.drop(delete_list, axis=0, inplace=False)
            # df.to_csv(data_path, index=False)

        return data_list

class GNNDataset4(InMemoryDataset):
    def __init__(self, root, index=0, types='train', transform=None, pre_transform=None, pre_filter=None):
        self.dataset = root.split("/")[1]
        self.root = root
        self.process()

    @property
    def raw_file_names(self):
        return [
            f'data/{self.dataset}/substitute.csv', 
            f'data/{self.dataset}/ChemDiv_noise.csv'
            ]

    @property
    def processed_file_names(self):
        return ['substitute.pt', 'ChemDiv_noise.pt']

    def download(self):
        pass


    def process(self):
        test_list = self.process_test_data(self.raw_file_names[0])
        noise_list = self.process_test_data(self.raw_file_names[1])
        print('Graph construction done. Saving to file.')

        # save preprocessed test data:
        data, slices = self.collate(test_list)
        torch.save((data, slices), f'{self.root}/processed/{self.processed_file_names[0]}')
        data, slices = self.collate(noise_list)
        torch.save((data, slices), f'{self.root}/processed/{self.processed_file_names[1]}')

    def process_test_data(self, data_path):
        # pdb.set_trace()
        df = pd.read_csv(data_path)
        data_list = []
        delete_list = []
        for i, row in tqdm(df.iterrows()):
            smi = row['compound_iso_smiles']
            sequence = row['target_sequence'].upper()
            affinity = row['affinity']
            protein_name = row['protein_name']
            mol = Chem.MolFromSmiles(smi)
            if mol == None:
                print("Unable to process: ", smi)
                continue
            x, edge_index, edge_attr = mol_to_graph_without_rm(mol)

            target = seqs2int(sequence)
            target_len = 1200
            if len(target) < target_len:
                target = np.pad(target, (0, target_len - len(target)))
            else:
                target = target[:target_len]

            data = DATA.Data(
                x=torch.FloatTensor(x),
                smi=smi,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.FloatTensor([affinity]),
                target=torch.LongTensor([target]),
                protein_name=protein_name
            )
            data_list.append(data)
            pdb.set_trace()

        # if len(delete_list) > 0:
            # df = df.drop(delete_list, axis=0, inplace=False)
            # df.to_csv(data_path, index=False)
        print('mol_to_graph_without_rm')
        return data_list

class GNNDataset_mamba(InMemoryDataset):
    def __init__(self, root, index=0, types='train', transform=None, pre_transform=None, pre_filter=None):
        # self.dataset = 'BindingDB_Kd_4S'
        self.dataset = root.split("/")[1]
        self.root = root
        self.process()

    @property
    def raw_file_names(self):
        return [
            f'data/{self.dataset}/train.csv', 
            f'data/{self.dataset}/test_S1.csv',
            f'data/{self.dataset}/test_S2.csv',
            f'data/{self.dataset}/test_S3.csv',
            f'data/{self.dataset}/test_S4.csv'
        ]

    @property
    def processed_file_names(self):
        return [
            f'{self.dataset}_mamba_train.pt', 
            f'{self.dataset}_mamba_test_S1.pt',
            f'{self.dataset}_mamba_test_S2.pt',
            f'{self.dataset}_mamba_test_S3.pt',
            f'{self.dataset}_mamba_test_S4.pt'
        ]

    def download(self):
        pass


    def process(self):
        train_list = self.process_train_data(self.raw_file_names[0])
        test_S1_list = self.process_test_data(self.raw_file_names[1])
        test_S2_list = self.process_test_data(self.raw_file_names[2])
        test_S3_list = self.process_test_data(self.raw_file_names[3])
        test_S4_list = self.process_test_data(self.raw_file_names[4])
        print('Graph construction done. Saving to file.')

        # save preprocessed train data:
        data, slices = self.collate(train_list)
        torch.save((data, slices), f'{self.root}/processed/{self.processed_file_names[0]}')

        # save preprocessed test data:
        data, slices = self.collate(test_S1_list)
        torch.save((data, slices), f'{self.root}/processed/{self.processed_file_names[1]}')
        data, slices = self.collate(test_S2_list)
        torch.save((data, slices), f'{self.root}/processed/{self.processed_file_names[2]}')
        data, slices = self.collate(test_S3_list)
        torch.save((data, slices), f'{self.root}/processed/{self.processed_file_names[3]}')
        data, slices = self.collate(test_S4_list)
        torch.save((data, slices), f'{self.root}/processed/{self.processed_file_names[4]}')

    def process_test_data(self, data_path):
        # pdb.set_trace()
        df = pd.read_csv(data_path)
        data_list = []
        delete_list = []
        for i, row in tqdm(df.iterrows()):
            smi = row['compound_iso_smiles']
            sequence = row['target_sequence'].upper()
            affinity = row['affinity']
            protein_name = row['protein_name']
            mol = Chem.MolFromSmiles(smi)
            if mol == None:
                print("Unable to process: ", smi)
                continue
            x, edge_index, edge_attr = mol_to_graph_without_rm(mol)

            target = seqs2int(sequence)
            target_len = 1200
            if len(target) < target_len:
                target = np.pad(target, (0, target_len - len(target)))
            else:
                target = target[:target_len]

            data = DATA.Data(
                x=torch.FloatTensor(x),
                smi=smi,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.FloatTensor([affinity]),
                target=torch.LongTensor([target]),
                protein_name=protein_name
            )

            transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')
            data = transform(data)

            data_list.append(data)

        # if len(delete_list) > 0:
            # df = df.drop(delete_list, axis=0, inplace=False)
            # df.to_csv(data_path, index=False)
        print('mol_to_graph_without_rm')
        return data_list

    def process_train_data(self, data_path):
        # pdb.set_trace()
        df = pd.read_csv(data_path)
        data_list = []
        delete_list = []
        for i, row in tqdm(df.iterrows()):
            smi = row['compound_iso_smiles']
            sequence = row['target_sequence'].upper()
            affinity = row['affinity']
            protein_name = row['protein_name']
            mol = Chem.MolFromSmiles(smi)
            if mol == None:
                print("Unable to process: ", smi)
                continue

            x_list, edge_index_list, edge_attr_list = mol_to_graph(mol, times = 2)

            target = seqs2int(sequence)
            target_len = 1200
            if len(target) < target_len:
                target = np.pad(target, (0, target_len - len(target)))
            else:
                target = target[:target_len]
            
            if x_list and edge_index_list:
                for x, edge_index, edge_attr in zip(x_list, edge_index_list, edge_attr_list):
                    try:
                        data = DATA.Data(
                            x=torch.FloatTensor(x),
                            smi=smi,
                            edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                            edge_attr=torch.FloatTensor(edge_attr),
                            y=torch.FloatTensor([affinity]),
                            target=torch.LongTensor([target]),
                            protein_name=protein_name
                        )

                        transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')
                        data = transform(data)

                        data_list.append(data)
                        # pdb.set_trace()
                    except Exception as e:
                        print(e)
                        continue

        print('process train data')
        # if len(delete_list) > 0:
            # df = df.drop(delete_list, axis=0, inplace=False)
            # df.to_csv(data_path, index=False)

        return data_list

class GNNDataset_noaug(InMemoryDataset):
    def __init__(self, root, index=0, types='train', transform=None, pre_transform=None, pre_filter=None):
        # self.dataset = 'BindingDB_Kd_4S'
        self.dataset = root.split("/")[1]
        self.root = root
        self.process()

    @property
    def raw_file_names(self):
        return [
            f'data/{self.dataset}/train200.csv', 
            f'data/{self.dataset}/test_S1.csv',
            f'data/{self.dataset}/test_S2.csv',
            f'data/{self.dataset}/test_S3.csv',
            f'data/{self.dataset}/test_S4.csv'
        ]

    @property
    def processed_file_names(self):
        return [
            f'{self.dataset}_train.pt', 
            f'{self.dataset}_test_S1.pt',
            f'{self.dataset}_test_S2.pt',
            f'{self.dataset}_test_S3.pt',
            f'{self.dataset}_test_S4.pt'
        ]

    def download(self):
        pass


    def process(self):
        train_list = self.process_test_data(self.raw_file_names[0])
        test_S1_list = self.process_test_data(self.raw_file_names[1])
        test_S2_list = self.process_test_data(self.raw_file_names[2])
        test_S3_list = self.process_test_data(self.raw_file_names[3])
        test_S4_list = self.process_test_data(self.raw_file_names[4])
        print('Graph construction done. Saving to file.')

        # save preprocessed train data:
        data, slices = self.collate(train_list)
        torch.save((data, slices), f'{self.root}/processed_noaug/{self.processed_file_names[0]}')

        # save preprocessed test data:
        data, slices = self.collate(test_S1_list)
        torch.save((data, slices), f'{self.root}/processed_noaug/{self.processed_file_names[1]}')
        data, slices = self.collate(test_S2_list)
        torch.save((data, slices), f'{self.root}/processed_noaug/{self.processed_file_names[2]}')
        data, slices = self.collate(test_S3_list)
        torch.save((data, slices), f'{self.root}/processed_noaug/{self.processed_file_names[3]}')
        data, slices = self.collate(test_S4_list)
        torch.save((data, slices), f'{self.root}/processed_noaug/{self.processed_file_names[4]}')

    def process_test_data(self, data_path):
        # pdb.set_trace()
        df = pd.read_csv(data_path)
        data_list = []
        delete_list = []
        for i, row in tqdm(df.iterrows()):
            smi = row['compound_iso_smiles']
            sequence = row['target_sequence'].upper()
            affinity = row['affinity']
            protein_name = row['protein_name']
            mol = Chem.MolFromSmiles(smi)
            if mol == None:
                print("Unable to process: ", smi)
                continue
            x, edge_index, edge_attr = mol_to_graph_without_rm(mol)

            target = seqs2int(sequence)
            target_len = 1200
            if len(target) < target_len:
                target = np.pad(target, (0, target_len - len(target)))
            else:
                target = target[:target_len]

            data = DATA.Data(
                x=torch.FloatTensor(x),
                smi=smi,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.FloatTensor([affinity]),
                target=torch.LongTensor([target]),
                protein_name=protein_name
            )
            data_list.append(data)

        # if len(delete_list) > 0:
            # df = df.drop(delete_list, axis=0, inplace=False)
            # df.to_csv(data_path, index=False)
        print('mol_to_graph_without_rm')
        return data_list

    def process_train_data(self, data_path):
        # pdb.set_trace()
        df = pd.read_csv(data_path)
        data_list = []
        delete_list = []
        for i, row in tqdm(df.iterrows()):
            smi = row['compound_iso_smiles']
            sequence = row['target_sequence'].upper()
            affinity = row['affinity']
            protein_name = row['protein_name']
            mol = Chem.MolFromSmiles(smi)
            if mol == None:
                print("Unable to process: ", smi)
                continue

            x_list, edge_index_list, edge_attr_list = mol_to_graph(mol, times = 2)

            target = seqs2int(sequence)
            target_len = 1200
            if len(target) < target_len:
                target = np.pad(target, (0, target_len - len(target)))
            else:
                target = target[:target_len]
            
            if x_list and edge_index_list:
                for x, edge_index, edge_attr in zip(x_list, edge_index_list, edge_attr_list):
                    try:
                        data = DATA.Data(
                            x=torch.FloatTensor(x),
                            smi=smi,
                            edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                            edge_attr=torch.FloatTensor(edge_attr),
                            y=torch.FloatTensor([affinity]),
                            target=torch.LongTensor([target]),
                            protein_name=protein_name
                        )
                        data_list.append(data)
                    except Exception as e:
                        print(e)
                        continue

        print('process train data')
        # if len(delete_list) > 0:
            # df = df.drop(delete_list, axis=0, inplace=False)
            # df.to_csv(data_path, index=False)

        return data_list

class GNNDataset_single(InMemoryDataset):
    def __init__(self, root, dataset, index=0, types='train', transform=None, pre_transform=None, pre_filter=None):
        self.dataset = dataset
        self.root = root
        self.process()

    @property
    def raw_file_names(self):
        return [f'{self.root}/{self.dataset}.csv']

    @property
    def processed_file_names(self):
        return [f'{self.dataset}.pt']

    def download(self):
        pass


    def process(self):
        test_list = self.process_test_data(self.raw_file_names[0])
        print('Graph construction done. Saving to file.')

        # save preprocessed test data:
        data, slices = self.collate(test_list)
        torch.save((data, slices), f'{self.root}/processed/{self.processed_file_names[0]}')

    def process_test_data(self, data_path):
        print(f"Reading {data_path}!")
        df = pd.read_csv(data_path)
        # pdb.set_trace()
        data_list = []
        delete_list = []
        for i, row in tqdm(df.iterrows()):
            smi = row['compound_iso_smiles']
            sequence = row['target_sequence'].upper()
            affinity = row['affinity']
            protein_name = row['protein_name']
            mol = Chem.MolFromSmiles(smi)
            if mol == None:
                print("Unable to process: ", smi)
                continue
            x, edge_index, edge_attr = mol_to_graph_without_rm(mol)

            target = seqs2int(sequence)
            target_len = 1200
            if len(target) < target_len:
                target = np.pad(target, (0, target_len - len(target)))
            else:
                target = target[:target_len]

            data = DATA.Data(
                x=torch.FloatTensor(x),
                smi=smi,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.FloatTensor([affinity]),
                target=torch.LongTensor([target]),
                protein_name=protein_name
            )
            data_list.append(data)

        # if len(delete_list) > 0:
            # df = df.drop(delete_list, axis=0, inplace=False)
            # df.to_csv(data_path, index=False)
        print('mol_to_graph_without_rm')
        return data_list

class GNNDataset_GPCRs(InMemoryDataset):
    def __init__(self, root, index=0, types='train', transform=None, pre_transform=None, pre_filter=None):
        self.dataset = root.split("/")[1]
        self.root = root
        self.process()

    @property
    def raw_file_names(self):
        return [f'data/{self.dataset}/train.csv', 
                f'data/{self.dataset}/test.csv',
                f'data/{self.dataset}/GPCRs.csv']

    @property
    def processed_file_names(self):
        return [f'{self.dataset}_train.pt', f'{self.dataset}_test.pt', f'{self.dataset}.pt']

    def download(self):
        pass


    def process(self):
        train_list = self.process_train_data(self.raw_file_names[0])
        test_list = self.process_test_data(self.raw_file_names[1])
        GPCRs_list = self.process_test_data(self.raw_file_names[2])
        print('Graph construction done. Saving to file.')

        # save preprocessed train data:
        data, slices = self.collate(train_list)
        torch.save((data, slices), f'{self.root}/processed/{self.processed_file_names[0]}')

        # save preprocessed test data:
        data, slices = self.collate(test_list)
        torch.save((data, slices), f'{self.root}/processed/{self.processed_file_names[1]}')
        data, slices = self.collate(GPCRs_list)
        torch.save((data, slices), f'{self.root}/processed/{self.processed_file_names[2]}')

    def process_test_data(self, data_path):
        # pdb.set_trace()
        df = pd.read_csv(data_path)
        data_list = []
        delete_list = []
        for i, row in tqdm(df.iterrows()):
            smi = row['compound_iso_smiles']
            sequence = row['target_sequence'].upper()
            affinity = row['affinity']
            protein_name = row['protein_name']
            mol = Chem.MolFromSmiles(smi)
            if mol == None:
                print("Unable to process: ", smi)
                continue
            x, edge_index, edge_attr = mol_to_graph_without_rm(mol)

            target = seqs2int(sequence)
            target_len = 1200
            if len(target) < target_len:
                target = np.pad(target, (0, target_len - len(target)))
            else:
                target = target[:target_len]

            data = DATA.Data(
                x=torch.FloatTensor(x),
                smi=smi,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.FloatTensor([affinity]),
                target=torch.LongTensor([target]),
                protein_name=protein_name
            )
            data_list.append(data)

        # if len(delete_list) > 0:
            # df = df.drop(delete_list, axis=0, inplace=False)
            # df.to_csv(data_path, index=False)
        print('mol_to_graph_without_rm')
        return data_list

    def process_train_data(self, data_path):
        # pdb.set_trace()
        df = pd.read_csv(data_path)
        data_list = []
        delete_list = []
        for i, row in tqdm(df.iterrows()):
            smi = row['compound_iso_smiles']
            sequence = row['target_sequence'].upper()
            affinity = row['affinity']
            protein_name = row['protein_name']
            mol = Chem.MolFromSmiles(smi)
            if mol == None:
                print("Unable to process: ", smi)
                continue

            x_list, edge_index_list, edge_attr_list = mol_to_graph(mol, times = 2)

            target = seqs2int(sequence)
            target_len = 1200
            if len(target) < target_len:
                target = np.pad(target, (0, target_len - len(target)))
            else:
                target = target[:target_len]
            
            if x_list and edge_index_list:
                for x, edge_index, edge_attr in zip(x_list, edge_index_list, edge_attr_list):
                    try:
                        data = DATA.Data(
                            x=torch.FloatTensor(x),
                            smi=smi,
                            edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                            edge_attr=torch.FloatTensor(edge_attr),
                            y=torch.FloatTensor([affinity]),
                            target=torch.LongTensor([target]),
                            protein_name=protein_name
                        )
                        data_list.append(data)
                        # pdb.set_trace()
                    except Exception as e:
                        print(e)
                        continue

        print('process train data')
        # if len(delete_list) > 0:
            # df = df.drop(delete_list, axis=0, inplace=False)
            # df.to_csv(data_path, index=False)

        return data_list
    
class GNNDataset_tvt(InMemoryDataset):
    def __init__(self, root, types='train', transform=None, pre_transform=None, pre_filter=None):
        self.dataset = root.split("/")[1]
        self.root = root
        self.process()

    @property
    def raw_file_names(self):
        return [f'data/{self.dataset}/{self.dataset}_train.csv', 
                f'data/{self.dataset}/{self.dataset}_test.csv',
                f'data/{self.dataset}/{self.dataset}_valid.csv']

    @property
    def processed_file_names(self):
        return [f'{self.dataset}_train.pt', f'{self.dataset}_test.pt', f'{self.dataset}_valid.pt']

    def download(self):
        pass


    def process(self):
        train_list = self.process_train_data(self.raw_file_names[0])
        test_list = self.process_test_data(self.raw_file_names[1])
        GPCRs_list = self.process_test_data(self.raw_file_names[2])
        print('Graph construction done. Saving to file.')

        # save preprocessed train data:
        data, slices = self.collate(train_list)
        torch.save((data, slices), f'{self.root}/processed/{self.processed_file_names[0]}')

        # save preprocessed test data:
        data, slices = self.collate(test_list)
        torch.save((data, slices), f'{self.root}/processed/{self.processed_file_names[1]}')
        data, slices = self.collate(GPCRs_list)
        torch.save((data, slices), f'{self.root}/processed/{self.processed_file_names[2]}')

    def process_test_data(self, data_path):
        # pdb.set_trace()
        df = pd.read_csv(data_path)
        data_list = []
        delete_list = []
        for i, row in tqdm(df.iterrows()):
            smi = row['compound_iso_smiles']
            sequence = row['target_sequence'].upper()
            affinity = row['affinity']
            protein_name = row['protein_name']
            mol = Chem.MolFromSmiles(smi)
            if mol == None:
                print("Unable to process: ", smi)
                continue
            x, edge_index, edge_attr = mol_to_graph_without_rm(mol)

            target = seqs2int(sequence)
            target_len = 1200
            if len(target) < target_len:
                target = np.pad(target, (0, target_len - len(target)))
            else:
                target = target[:target_len]

            data = DATA.Data(
                x=torch.FloatTensor(x),
                smi=smi,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.FloatTensor([affinity]),
                target=torch.LongTensor([target]),
                protein_name=protein_name
            )
            data_list.append(data)

        # if len(delete_list) > 0:
            # df = df.drop(delete_list, axis=0, inplace=False)
            # df.to_csv(data_path, index=False)
        print('mol_to_graph_without_rm')
        return data_list

    def process_train_data(self, data_path):
        # pdb.set_trace()
        df = pd.read_csv(data_path)
        data_list = []
        delete_list = []
        for i, row in tqdm(df.iterrows()):
            smi = row['compound_iso_smiles']
            sequence = row['target_sequence'].upper()
            affinity = row['affinity']
            protein_name = row['protein_name']
            mol = Chem.MolFromSmiles(smi)
            if mol == None:
                print("Unable to process: ", smi)
                continue

            x_list, edge_index_list, edge_attr_list = mol_to_graph(mol, times = 2)

            target = seqs2int(sequence)
            target_len = 1200
            if len(target) < target_len:
                target = np.pad(target, (0, target_len - len(target)))
            else:
                target = target[:target_len]
            
            if x_list and edge_index_list:
                for x, edge_index, edge_attr in zip(x_list, edge_index_list, edge_attr_list):
                    try:
                        data = DATA.Data(
                            x=torch.FloatTensor(x),
                            smi=smi,
                            edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                            edge_attr=torch.FloatTensor(edge_attr),
                            y=torch.FloatTensor([affinity]),
                            target=torch.LongTensor([target]),
                            protein_name=protein_name
                        )
                        data_list.append(data)
                        # pdb.set_trace()
                    except Exception as e:
                        print(e)
                        continue

        print('process train data')
        # if len(delete_list) > 0:
            # df = df.drop(delete_list, axis=0, inplace=False)
            # df.to_csv(data_path, index=False)

        return data_list
    


def debug():
    smi = 'Cc1cc(OCC2CCC(C(=O)O)C2)c(NC(=O)Nc2cnc(C#N)cn2)cc1Cl'
    mol = Chem.MolFromSmiles(smi)
    if mol == None:
        print("Unable to process: ", smi)
    x, edge_index, edge_attr = mol_to_graph_without_rm(mol)
    x_list, edge_index_list, edge_attr_list, removed_list = mol_to_graph(mol)
    data_list = []
    data = DATA.Data(
        x=torch.FloatTensor(x),
        smi=smi,
        edge_index=torch.LongTensor(edge_index),
        edge_attr=torch.FloatTensor(edge_attr),
        # y=torch.FloatTensor([affinity]),
        # target=torch.LongTensor([target]),
        # protein_name=protein_name
    )
    data_list.append(data)
    for x, edge_index, edge_attr in zip(x_list, edge_index_list, edge_attr_list):
        try:
            data = DATA.Data(
                x=torch.FloatTensor(x),
                smi=smi,
                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                edge_attr=torch.FloatTensor(edge_attr),
                # y=torch.FloatTensor([affinity]),
                # target=torch.LongTensor([target]),
                # protein_name=protein_name
            )
            data_list.append(data)
        except Exception as e:
            print(e)
            continue
    # aa = []
    # for ii in range(2):
    #     cnt = 0
    #     for jj in range(mol.GetNumAtoms()-len(removed_list[ii])):
    #         while jj+cnt in removed_list[ii]:
    #             cnt += 1
    #         aa.append((data_list[0].x[jj+cnt]==data_list[ii+1].x[jj]).all())
    #     pdb.set_trace()
    #     aa = []
    device = torch.device('cuda:1')
    model = ColdDTA().to(device)
    for data in data_list:
        data = data.to(device)
        pdb.set_trace()
        model(data)

        

if __name__ == "__main__":
    # GNNDataset('data/sample_cold_drug_davis_five_cross')
    # GNNDataset('data/cold_drug_davis_five_cross')
    # GNNDataset('data/cold_protein_davis_five_cross')
    # GNNDataset('data/cold_pair_davis_five_cross')
    # GNNDataset2('data/GPCRs')
    # GNNDataset2('data/kiba_wo_chem')
    # GNNDataset3('data/BindingDB_Kd_4S')
    # GNNDataset4('data/kiba+chem')
    # GNNDataset_mamba('data/kiba_4S')
    # GNNDataset_noaug('data/BindingDB_Kd_4S')
    GNNDataset_single('data/kiba+chem', 'merge_data2')
    # GNNDataset_GPCRs('data/GPCRs', 'GPCRs')
    # GNNDataset_tvt('data/kiba')
    # debug()
