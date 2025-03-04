import numpy as np
import pandas as pd

import rdkit
from rdkit import Chem
from rdkit.Chem import Draw, PeriodicTable

from torch_geometric.data.data import Data

def graph_from_smiles(smiles, ff,
                     ff_keys = ["epsilon","sigma","charge",
                               "neighbors charge"
                               ]):
    """
    iterate database!!
    
    1) if n_atoms <= xxx
    2) build atom and bond list and atom names
    3) check names vs database i.e. force field!
    4) map FF (M, sigma, eps, ch)
    5) build graph
    
    """
    
    m = Chem.MolFromSmiles(smiles)
    m = Chem.RemoveHs(m)
    
    pse = Chem.GetPeriodicTable()
    m_H = pse.GetAtomicWeight("H")
    
    """
    needed:
    - atomic symbol --> mapping
    - do the mapping here to get a feature matrix
    - bond list
    
    """
    
    no_atoms = 0
    for i,a in enumerate(m.GetAtoms()):
        no_atoms += 1
    
    m_Hs = Chem.AddHs(m)
    bond_list = []
    #atom_names = np.zeros(no_atoms, dtype=object)
    atom_names = np.array([""]*no_atoms, dtype=object)
    atom_neighbor_no = np.zeros(no_atoms)
    atom_weights = np.zeros(no_atoms)

    """
    Embedding:
    sig, eps, q, mass, ring no, bond_no, qmask
    """

    atom_embeddings = np.zeros([no_atoms,9])
        
    for i,a in enumerate(m.GetAtoms()):
        ii = a.GetAtomicNum()
        sym = a.GetSymbol()
        if sym != "C":
            # if not C, set charge to zero
            atom_embeddings[i,2] = 0  
        wei = pse.GetAtomicWeight(ii)
        #print(i,ii,sym, wei)
        #a.GetBonds()
        ### a.GetIsAromatic()
        # no of neighbors
        # add letter if:
        # - one neighbor (end)
        # - three or more neighbors (branch)
        ns = a.GetDegree()
        Hs = m_Hs.GetAtomWithIdx(i).GetDegree() - ns
        if Hs > 0:
            #print("modify symbol",Hs)
            if Hs==1:
                sym += "H"
            else:
                sym += "H" + str(int(Hs))
            wei += m_H*Hs
        ring_size = 0
        if a.GetIsAromatic():
            sym = "r"+sym
            for rr in range(12):
                if a.IsInRingSize(rr):
                    ring_size = rr
        atom_names[i] = sym
        atom_weights[i] = wei
        bond_no = 0
        mff = ff[ (ff["(pseudo)atom"]==sym ) & ( ff["neighbors"]==ns) ]
        embedding = mff[ff_keys[:-1]]
        #print(sym, embedding, bond_no)
        for b in a.GetBonds():
            # build bond list
            b.GetEndAtomIdx() 
            b.GetBeginAtomIdx()
            bond = np.sort([b.GetBeginAtomIdx(),b.GetEndAtomIdx()])
            bond_list.append( bond )
            # check bond type
            # SINGLE, DOUBLE, AROMATIC :)
            # b.GetBondType()
            bond_no += 1
            ### b.GetIsAromatic()
            neighbor_charge = np.squeeze(mff[ff_keys[-1]])
            if neighbor_charge != 0:
                iii = bond[np.where( np.array(bond) != i)]
                if atom_names[iii][0] == "C" or not atom_names[iii][0]:
                    atom_embeddings[iii,2] += neighbor_charge   
        #print(sym,embedding, bond_no)
        atom_neighbor_no[i] = bond_no
        atom_embeddings[i,:3] += embedding   
        atom_embeddings[i,3] = wei
        atom_embeddings[i,4] = ring_size
        atom_embeddings[i,5] = bond_no
        atom_embeddings[i,6] = Hs
        atom_embeddings[i,7] = a.GetAtomicNum()
    
    bond_list = np.unique(bond_list, axis=0)
    charge_mask = np.zeros(no_atoms)
    charge_mask[atom_embeddings[:,2] > 0] = 1
    charge_mask[atom_embeddings[:,2] < 0] =-1
    atom_embeddings[:,8] = charge_mask

    embedding_names = ["sigma","epsilon","charge",
                       "atom_mass","ring_size",
                       "bond_no","H_no","atomic_num",
                       "charge_mask"
                      ]
    return atom_embeddings, bond_list