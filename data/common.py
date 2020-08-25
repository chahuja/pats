import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
from pathlib import Path
import warnings
import pandas as pd
import numpy as np
import pdb

from argsUtils import *
from tqdm import tqdm

class HDF5():
  def __init__(self):
    pass
  
  '''
  Create a file if it does not exist, else appends data to key
  '''
  @staticmethod
  def append(filename, key, data):
    h5 = HDF5.h5_open(filename, 'a')
    try:
      HDF5.update_dataset(h5, key, data)
    except:
      #pdb.set_trace()
      warnings.warn('could not update dataset {} with filename {}'.format(key, filename))
    HDF5.h5_close(h5)

  @staticmethod
  def load(filename, key):
    h5 = HDF5.h5_open(filename, 'r')
    data = h5[key]
    return data, h5

  @staticmethod
  def isDatasetInFile(filename, key):
    h5 = HDF5.h5_open(filename, 'r')
    if key in h5:
      h5.close()
      return True
    else:
      h5.close()
      return False
      
  @staticmethod
  def h5_open(filename, mode):
    ## create the parent directory if does not exist
    os.makedirs(Path(filename).parent, exist_ok=True)
    return h5py.File(filename, mode)

  @staticmethod
  def h5_close(h5):
    h5.close()

  @staticmethod
  def add_dataset(h5, key, data, exist_ok=False):
    if key in h5:
      if exist_ok:
        warnings.warn('dataset {} already exists. Updating data...'.format(key))
        del h5[key]
        h5.create_dataset(key, data=data)
      else:
        warnings.warn('dataset {} already exists. Skipping...'.format(key))
    else:
      h5.create_dataset(key, data=data)

  @staticmethod
  def update_dataset(h5, key, data):
    HDF5.add_dataset(h5, key, data, exist_ok=True)

  '''
  Delete a dataset in an hdf file
  
  Arguments
    h5: file pointer to the hdf file
    key: key to be deleted

  Return 
    ``True`` if key found and deleted
    ``False`` if key not found
  '''
  @staticmethod
  def del_dataset(h5, key):
    if key in h5:
      del h5[key]
      return True
    else:
      warnings.warn('Key not found. Skipping...')
      return False

  @staticmethod
  def add_key(base_key, sub_keys=[]):
    if isinstance(sub_keys, str):
      sub_keys = [sub_keys]

    sub_keys = '/'.join(sub_keys)
    new_key = (Path(base_key)/Path(sub_keys)).as_posix()
    return new_key

  
class Modality(HDF5):
  def __init__(self, path2data='../dataset/groot/data',
               path2outdata='../dataset/groot/data',
               speaker='all',
               preprocess_methods=['log_mel']):
    super(Modality, self).__init__()
    self.path2data = path2data
    self.df = pd.read_csv(Path(self.path2data)/'cmu_intervals_df.csv', dtype=object)
    self.df.loc[:, 'delta_time'] = self.df['delta_time'].apply(float)
    self.df.loc[:, 'interval_id'] = self.df['interval_id'].apply(str)
    
    self.path2outdata = path2outdata
    self.speaker = speaker
    self.preprocess_methods = preprocess_methods

  def preprocess(self):
    raise NotImplementedError

  def del_keys(self, h5_key):
    if self.speaker != 'all':
      speakers = [self.speaker]
    else:
      speakers = self.speakers

    for speaker in tqdm(speakers, desc='speakers', leave=False):
      tqdm.write('Speaker: {}'.format(speaker))
      df_speaker = self.get_df_subset(speaker)
      interval_ids = df_speaker['interval_id'].unique()
      for preprocess_method in self.preprocess_methods:
        for interval_id in tqdm(interval_ids, desc='intervals'):
          filename = Path(self.path2outdata)/'processed'/speaker/'{}.h5'.format(interval_id)
          key = self.add_key(h5_key[0], [preprocess_method])

          ## delete dataset
          h5 = self.h5_open(filename.as_posix(), 'a')
          key_flag = self.del_dataset(h5, key)
          if not key_flag:
            break ## ignore files of a speaker if the first file does not have ``key``
          self.h5_close(h5)

  def get_df_subset(self, column, value):
    if isinstance(value, list):
      return self.df[self.df[column].isin(value)]
    else:
      return self.df[self.df[column] == value]
  
  @property
  def speakers(self):
    return [
      'oliver', #TV sitting high_freq
      'jon', #TV sitting 
      'conan', #TV standing high_freq
      'rock', #lec sitting
      'chemistry', #lec sitting
      'ellen', #TV standing
      'almaram', #eval sitting
      'angelica', #eval sitting
      'seth', #TV sitting low frequency
      'shelly', #TV sitting
      'colbert', #TV standing high_freq
      'corden', #TV standing 
      'fallon', #TV standing
      'huckabee', #TV standing
      'maher', #TV standing
      'lec_cosmic', #lec sitting
      'lec_evol', #lec sitting
      'lec_hist', #lec sitting
      'lec_law', #lec sitting
      'minhaj', #TV standing
      'ytch_charisma', #yt sitting
      'ytch_dating', #yt sitting
      'ytch_prof', #yt sitting
      'bee', #TV standing
      'noah' #TV sitting
    ]

  @property
  def inv_speakers(self):
    dc = {}
    for i, speaker in enumerate(self.speakers):
      dc[speaker] = i
    return dc
  
  def speaker_id(self, speaker):
    return self.inv_speakers[speaker]

class MissingData(HDF5):
  def __init__(self, path2data):
    super(MissingData, self).__init__()
    self.path2file = Path(path2data)/'missing_intervals.h5'
    if not os.path.exists(self.path2file):
      h5 = HDF5.h5_open(self.path2file, 'a')
      HDF5.h5_close(h5)
    self.key = 'intervals'
    self.missing_data_list = []

  def append_interval(self, data):
    self.missing_data_list.append(data)
    warnings.warn('interval_id: {} not found.'.format(data))
    
  def save_intervals(self, missing_data_list):
    '''
    Append `data` to the missing_intervals.h5 file
    '''
    dt = h5py.special_dtype(vlen=str)
    if HDF5.isDatasetInFile(self.path2file, self.key):
      intervals, h5 = HDF5.load(self.path2file, self.key)
      intervals = set(intervals[()])
      h5.close()
      intervals.update(set(missing_data_list) - {None})
      intervals = np.array(list(intervals), dtype=dt)
    else:
      intervals = np.array(list(set(missing_data_list) - {None}), dtype=dt)

    HDF5.append(self.path2file, self.key, intervals)

  def save(self, missing_data_list):
    '''
    Add new missing `data` from the current set to the missing_intervals.h5 file
    '''
    dt = h5py.special_dtype(vlen=str)
    intervals = np.array(list(set(missing_data_list) - {None}), dtype=dt)
    HDF5.append(self.path2file, self.key, intervals)

  def load_intervals(self):
    if HDF5.isDatasetInFile(self.path2file, self.key):
      intervals, h5 = HDF5.load(self.path2file, self.key)
      intervals = set(intervals[()])
      h5.close()
    else:
      intervals = set()
    return intervals

'''
Delete keys from the processed dataset stored in hdf files

path2data: Irrelevant
path2outdata: path to processed data 
speaker: 'all' or a particular speaker
preprocess_methods: list of preprocess_methods to delete
modalities: modality to delete. 
            deleting keys of different modalities must be deleted separately
            eg: 'audio', 'pose' etc.
'''
def delete_keys(args, exp_num):
  modality = Modality(args.path2data, args.path2outdata,
                      args.speaker, args.preprocess_methods)
  modality.del_keys(args.modalities)

if __name__ == '__main__':
  argparseNloop(delete_keys)
