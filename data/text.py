import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import numpy as np
import pandas as pd
import pdb
from tqdm import tqdm
import warnings
import h5py

from pycasper.pathUtils import replace_Nth_parent
from common import Modality, MissingData, HDF5

import warnings

import torch
from torch.utils.data._utils.collate import default_collate
from functools import partial

def pad(datasets, key, dim):
  sizes = []
  for data in datasets:
    data = data[key]
    sizes.append(data.shape[dim])
  max_length = max(sizes)
  new_datasets = []
  lengths = []
  for data in datasets:
    data = data[key]
    length = data.shape[dim]
    zero_shape = list(data.shape)
    zero_shape[dim] = max_length-length
    new_datasets.append(np.concatenate([data, np.zeros(zero_shape)], axis=dim))
    lengths.append(length)
  return default_collate(new_datasets), default_collate(lengths)

def collate_fn_pad(batch, pad_key='text/meta', dim=0):
  if isinstance(batch[0], dict):
    data_dict = {}
    for key in batch[0]:
      if key in pad_key:
        padded_outs = pad(batch, key, dim=dim)
        if key == pad_key[-1]: ## TODO hardcoded to use the last key which is text/token_duration
          data_dict[key], data_dict['text/token_count'] = padded_outs[0], padded_outs[1]
        else:
          data_dict[key] = padded_outs[0]
      else:
        data_dict[key] = default_collate([d[key] for d in batch])
    return data_dict
  else:
    return default_collate(batch)

class Text(Modality):
  def __init__(self, path2data='../dataset/groot/data',
               path2outdata='../dataset/groot/data',
               speaker='all',
               preprocess_methods=['w2v'],
               text_aligned=0):
    super(Text, self).__init__(path2data=path2data)
    self.path2data = path2data
    self.df = pd.read_csv(Path(self.path2data)/'cmu_intervals_df.csv', dtype=object)
    self.df.loc[:, 'delta_time'] = self.df['delta_time'].apply(float)
    self.df.loc[:, 'interval_id'] = self.df['interval_id'].apply(str)
    
    self.path2outdata = path2outdata
    self.speaker = speaker
    self.preprocess_methods = preprocess_methods

    self.missing = MissingData(self.path2data)

    ## list of word2-vec models
    self.w2v_models = []
    self.text_aligned = text_aligned
    

  def fs(self, modality):
    return 15
  
  @property
  def h5_key(self):
    return 'text'
