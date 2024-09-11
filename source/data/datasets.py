import h5py
import numpy as np

from .constants import LABEL_SIZES
from torch.utils.data import Dataset


class HarmonyBassChromaDataset(Dataset):
    def __init__(self, datapath: str, split: str):
        super().__init__()
        assert split in ['train', 'valid', 'test'], 'split must be in ["train", "valid", "test"]'
        
        self.split = split
        self.datapath = datapath
        with h5py.File(self.datapath, 'r') as fp:
            self.keys = list(fp[self.split].keys())

    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, idx):
        key = self.keys[idx]
        
        with h5py.File(self.datapath, 'r') as fp:
            spec = fp[self.split][key]['spec'][:]            
            bass_chroma = fp[self.split][key]['bass_chroma'][:]            
            harmony_chroma = fp[self.split][key]['harmony_chroma'][:]

        return spec, bass_chroma, harmony_chroma


class FunctionalHarmonyDataset(Dataset):
    def __init__(self, datapath: str, split: str, use_harmony_only_chroma: bool = True):
        super().__init__()
        assert split in ['train', 'valid', 'test'], 'split must be in ["train", "valid", "test"]'

        self.split = split
        self.datapath = datapath
        self.use_harmony_only_chroma = use_harmony_only_chroma
        
        with h5py.File(self.datapath, 'r') as fp:
            self.keys = list(fp['functional_harmony'][self.split].keys())

    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, idx):
        key = self.keys[idx]
        
        with h5py.File(self.datapath, 'r') as fp:
            if self.use_harmony_only_chroma:
                features = fp['functional_harmony'][f'{self.split}/{key}/madmom_chroma'][:]
            else:
                bass_chroma = fp['functional_harmony'][f'{self.split}/{key}/bass_chroma'][:]
                harmony_chroma = fp['functional_harmony'][f'{self.split}/{key}/harmony_chroma'][:]
                features = np.concatenate([bass_chroma, harmony_chroma], axis=0)
            
            label_indices = fp['functional_harmony'][f'{self.split}/{key}/label_indices'][:]

            tivs_information = {
                'madmom_key_similarities': fp['functional_harmony'][self.split][key]['madmom_key_similarities'][:].astype(np.float32),
                'harmonybass_key_similarities': fp['functional_harmony'][self.split][key]['harmonybass_key_similarities'][:].astype(np.float32)
            }

            # Sometimes NaN values may appear due to resampling problems
            tivs_information['madmom_key_similarities'] = np.nan_to_num(tivs_information['madmom_key_similarities'])
            tivs_information['harmonybass_key_similarities'] = np.nan_to_num(tivs_information['harmonybass_key_similarities'])
        
        # Converting label matrix to one hot list
        label_one_hot = []
        for i, num_classes in enumerate(LABEL_SIZES):
            one_hot = self.one_hot(label_indices[i], num_classes)
            label_one_hot.append(one_hot)
            
        return features, label_indices, label_one_hot, tivs_information

    @classmethod
    def one_hot(cls, a, num_classes):
        return np.squeeze(np.eye(num_classes)[a.reshape(-1)]).astype(np.float32)
