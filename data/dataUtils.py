import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm
from pathlib import Path
import time
import pdb
from joblib import Parallel, delayed
import numpy as np
from nltk.corpus import stopwords

from skeleton import *
from audio import *
from text import *
from common import *
from argsUtils import *
import bisect

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Sampler
from torch.utils.data._utils.collate import default_collate
from transformers import BertTokenizer
import logging
logging.getLogger('transformers').setLevel(logging.CRITICAL)

from functools import partial

class DummyData(Dataset):
  def __init__(self, variable_list=['pose', 'audio'], length=1000, random=False, pause=False):
    super(DummyData, self).__init__()
    self.variable_list = variable_list
    self.len = length
    self.pause = pause
    
    if random:
      self.data = {variable:torch.rand(self.len, 30, 50) + 1 for variable in self.variable_list}
    else:
      self.data = {variable:torch.arange(self.len) + 1 for variable in self.variable_list}
    
  def __getitem__(self, idx):
    if self.pause:
      time.sleep(self.pause)
    return {variable:self.data[variable][idx].to(torch.double) for variable in self.variable_list}
  
  def __len__(self):
    return self.len


class Data(Modality):
  r'''
  Wrapper for DataLoaders

  Arguments:
    path2data (str): path to dataset.
    speaker (str): speaker name. 
    modalities (list of str): list of modalities to wrap in the dataloader. These modalities are basically keys of the hdf5 files which were preprocessed earlier (default: ``['pose/data', 'audio/log_mel']``)
    fs_new (list, optional): new frequency of modalities, to which the data is up/downsampled to. (default: ``[15, 15]``).
    time (float, optional): time snippet length in seconds. (default: ``4.3``).
    split (tuple or None, optional): split fraction of train and dev sets. Must add up to less than 1. If ``None``, use ``dataset`` columns in the master dataframe (loaded in self.df) to decide train, dev and test split. (default: ``None``).
    batch_size (int, optional): batch size of the dataloader. (default: ``100``).
    shuffle (boolean, optional): set to ``True`` to have the data reshuffled at every epoch (default: ``False``).
    num_workers (int, optional): set to values >0 to have more workers to load the data. argument for torch.utils.data.DataLoader. (default: ``15``). 

  Example::
    from data.dataUtils import Data 
    data = Data('../dataset/groot/data/', 'oliver', ['pose/data'], [15])

    for batch in data.train:
      break

    print(batch).

  '''
  def __init__(self, path2data, speaker,
               modalities = ['pose/data', 'audio/log_mel_512'],
               fs_new=[15, 15], time=4.3, 
               split=None,
               batch_size=100, shuffle=True, num_workers=0,
               window_hop=0,
               load_data=True,
               style_iters=0,
               num_training_sample=None,
               sample_all_styles=0,
               repeat_text=1,
               quantile_sample=None,
               quantile_num_training_sample=None,
               weighted=0,
               filler=0,
               num_training_iters=None):
    super().__init__(path2data=path2data)
    self.path2data = path2data
    self.speaker = speaker
    self.modalities = modalities
    self.fs_new = fs_new
    self.time = time
    self.split = split
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.num_workers = num_workers
    self.window_hop = window_hop
    self.load_data = load_data
    self.style_iters = style_iters ## used to call a train sampler
    self.num_training_sample = num_training_sample
    self.sample_all_styles = sample_all_styles
    self.repeat_text = repeat_text
    self.quantile_sample = quantile_sample
    self.quantile_num_training_sample = quantile_num_training_sample
    self.weighted = weighted
    self.filler = filler
    if self.filler:
      self.stopwords = stopwords.words('english')
      self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    else:
      self.stopwords, self.tokenizer = None, None
    self.num_training_iters = num_training_iters
      
    self.text_in_modalities = False
    for modality in self.modalities:
      if 'text' in modality:
        self.text_in_modalities = True
    
    self.missing = MissingData(self.path2data)

    if isinstance(self.speaker, str):
      self.speaker = [self.speaker]
    
    ## Load all modalities
    self.modality_classes = self._load_modality_classes()
    
    ## Load the master table
    self.df = pd.read_csv((Path(self.path2data)/'cmu_intervals_df.csv').as_posix())
    self.df = self.df.append(pd.read_csv((Path(self.path2data)/'cmu_intervals_df_transforms.csv').as_posix())) ## file with evil twins
    self.df.loc[:, 'interval_id'] = self.df['interval_id'].apply(str)

    ## Check for missing_data
    #self.missing_data
    
    if speaker[0] == 'all':
#      self.df = self.get_df_subset('speaker', self.speakers)
      self.speaker = self.speakers
    else:
      pass
#      self.df = self.get_df_subset('speaker', speaker)

    self.df = self.get_df_subset('speaker', self.speaker)
    ## Create Style Dictionary
    self.style_dict = {sp:i for i, sp in enumerate(self.speaker)}

    assert len(self.df.values), 'speaker `{}` not found'.format(speaker)

    #if self.load_data:
    ## get train-dev-test split
    self.datasets = self.tdt_split()
    self.dataLoader_kwargs = {'batch_size':batch_size,
                              'shuffle':shuffle,
                              'num_workers':num_workers,
                              'pin_memory':False}

    ## if not repeat_text, do not repeat the word vectors to match the fs
    #if True: #not self.repeat_text:
    if self.text_in_modalities:
      ## always keep text/token_duration at the end to comply with the collate_fn_pad
      pad_keys = ['text/w2v', 'text/bert', 'text/filler', 'text/tokens', 'text/token_duration']
      self.dataLoader_kwargs.update({'collate_fn':partial(collate_fn_pad, pad_key=pad_keys, dim=0)})
      
    self.update_dataloaders(time, window_hop)
      
  def _load_modality_classes(self):
    modality_map = {}
    for modality in self.modalities:
      mod = modality.split('/')[0]
      modality_map[modality] = self.mod_map(mod)

    return modality_map

  def mod_map(self, mod):
    mod_map_dict = {
      'pose': Skeleton2D,
      'audio': Audio,
      'text': Text
    }
    return mod_map_dict[mod](path2data=self.path2data, speaker=self.speaker)

  # def get_df_subset(self, column, value):
  #   if isinstance(value, list):
  #     return self.df[self.df[column].isin(value)]
  #   else:
  #     return self.df[self.df[column] == value]

  def getSpeaker(self, x):
    return self.get_df_subset('interval_id', x)['speaker'].values[0]

  def getPath2file(self, x):
    return (Path(self.path2data)/'processed'/self.getSpeaker(x)/str(x)).as_posix() + '.h5'

  def getStyle(self, interval_id):
    df_subset = self.get_df_subset('interval_id', interval_id)
    speaker = df_subset.speaker.iloc[0]
    try:
      style = self.style_dict[speaker]
    except:
      raise 'speaker style for {} not found'.format(speaker)
    return style

  def get_transforms_missing_intervals(self, missing_intervals):
    transforms = []
    for speaker in self.speaker:
      if '|' in speaker:
        transforms.append(speaker.split('|')[-1])

    transforms = sorted(list(set(transforms)))
    new_missing_intervals = set()
    for transform in transforms:
      for interval in missing_intervals:
        new_missing_intervals.update({'{}|{}'.format(interval, transform)})

    missing_intervals.update(new_missing_intervals)
    return missing_intervals

  def order_intervals(self, intervals):
    interval_dict = {i:[] for i in self.style_dict}
    for interval in intervals:
      interval_dict[self.getSpeaker(interval)].append(interval)
    intervals_dict = [(k, interval_dict[k]) for k in interval_dict]
    ordered_intervals = []
    for tup in intervals_dict:
      ordered_intervals += tup[1]
    return intervals_dict, ordered_intervals

  @property
  def minidataKwargs(self):
    minidataKwargs = {'modalities':self.modalities,
                      'fs_new':self.fs_new,
                      'time':self.time,
                      'modality_classes':self.modality_classes,
                      'window_hop':self.window_hop,
                      'repeat_text':self.repeat_text,
                      'text_in_modalities':self.text_in_modalities,
                      'filler':self.filler,
                      'stopwords':self.stopwords,
                      'tokenizer':self.tokenizer}      
    return minidataKwargs
  
  def get_minidata_list(self, intervals):
    return [MiniData(self.getPath2file(interval_id), style=self.getStyle(interval_id), **self.minidataKwargs)
                                   for interval_id in tqdm(intervals)]
  
  def tdt_split(self):
    if not self.split:
      df_train = self.get_df_subset('dataset', 'train')
      df_dev = self.get_df_subset('dataset', 'dev')
      df_test = self.get_df_subset('dataset', 'test')
    else:
      length = self.df.shape[0]
      end_train = int(length*self.split[0])
      start_dev = end_train
      end_dev = int(start_dev + length*self.split[1])
      start_test = end_dev

      df_train = self.df[:end_train]
      df_dev = self.df[start_dev:end_dev]
      df_test = self.df[start_test:]

    ## get missing intervals
    missing_intervals = self.missing.load_intervals()
    missing_intervals = self.get_transforms_missing_intervals(missing_intervals)
    
    ## get new train/dev/test intervals
    get_intervals = lambda x: sorted(list(set(x['interval_id'].unique()) - missing_intervals))
    train_intervals = get_intervals(df_train)
    dev_intervals = get_intervals(df_dev)
    test_intervals = get_intervals(df_test)

    self.train_intervals_all = train_intervals
    self.dev_intervals_all = dev_intervals
    self.test_intervals_all = test_intervals
    
    if not self.load_data: ## load a sample of the data just to get the shape
      train_intervals = train_intervals[:10]
      dev_intervals = dev_intervals[:10]
      test_intervals = test_intervals[:10]

    ## update_train_intervals
    train_intervals, dev_intervals, test_intervals, train_intervals_dict = self.update_intervals(train_intervals, dev_intervals, test_intervals)
    
    self.train_intervals = train_intervals
    self.dev_intervals = dev_intervals
    self.test_intervals = test_intervals
      
    dataset_train = ConcatDatasetIndex(self.get_minidata_list(train_intervals))
    dataset_dev = ConcatDatasetIndex(self.get_minidata_list(dev_intervals))
    dataset_test = ConcatDatasetIndex(self.get_minidata_list(test_intervals))

    self.dataset_train = dataset_train
    self.train_intervals_dict = train_intervals_dict
    self.train_sampler = self.get_train_sampler(dataset_train, train_intervals_dict)
    
    return {'train': dataset_train,
            'dev': dataset_dev,
            'test': dataset_test}
  
  def update_dataloaders(self, time, window_hop):
    ## update idx_list for all minidata
    for key in self.datasets:
      for d_ in self.datasets[key].datasets:
        d_.update_idx_list(time, window_hop)

    train_dataLoader_kwargs = self.dataLoader_kwargs.copy()
    if self.train_sampler:
      train_dataLoader_kwargs['shuffle'] = False
      
    self.train = DataLoader(ConcatDatasetIndex(self.datasets['train'].datasets), sampler=self.train_sampler, **train_dataLoader_kwargs)
    self.dev = DataLoader(ConcatDatasetIndex(self.datasets['dev'].datasets), **self.dataLoader_kwargs)
    self.test = DataLoader(ConcatDatasetIndex(self.datasets['test'].datasets), **self.dataLoader_kwargs)

  def get_alternate_class_sampler(self, dataset, intervals_dict, num_samples):
    class_count = []
    interval_offset = 0
    for tup in intervals_dict:
      count = 0
      for i in range(len(tup[1])):
        count+=len(dataset.datasets[i+interval_offset])
      class_count.append(count)
      interval_offset += len(tup[1])

    return AlternateClassSampler(class_count, num_samples*self.batch_size)

  def update_intervals(self, train_intervals, dev_intervals, test_intervals):
    def subsample_intervals(x):
      temp = []
      for x_ in x:
        if self.sample_all_styles > 0:
          temp.extend(x_[1][:self.sample_all_styles])
        elif self.sample_all_styles == -1:
          temp.extend(x_[1])
      return temp
      
    if self.sample_all_styles != 0:
      train_intervals_dict, train_intervals = self.order_intervals(train_intervals)
      dev_intervals_dict, dev_intervals = self.order_intervals(dev_intervals)
      test_intervals_dict, test_intervals = self.order_intervals(test_intervals)
      train_intervals = subsample_intervals(train_intervals_dict)
      dev_intervals = subsample_intervals(dev_intervals_dict)
      test_intervals = subsample_intervals(test_intervals_dict)
    elif self.style_iters > 0:  ## using AlternateClassSampler
      train_intervals_dict, train_intervals = self.order_intervals(train_intervals)
    else:
      train_intervals_dict = None
    return train_intervals, dev_intervals, test_intervals, train_intervals_dict

  def get_quantile_sample(self, data, q):
    pose_modality = None
    for key in self.modalities:
      if 'pose' in key:
        pose_modality = key
        break
    assert pose_modality is not None, "can't find pose modality"
    if isinstance(q, float) or isinstance(q, int):
      if q<1:
        kind = 'above'
      elif q>1:
        kind = 'rebalance'
        q = int(q)
      else:
        raise 'q can\'t be 1 or negative'
    elif isinstance(q, list):
      assert np.array([q_<=1 and q_>=0 for q_ in q]).all(), 'quantile_sample is in [0,1]'
      assert len(q) == 2
      kind = 'tail'
    
    ## get distribution of velocities
    diff = lambda x, idx: x[1:, idx] - x[:-1, idx]
    vel = lambda x, idx: (((diff(x, idx))**2).sum(-1)**0.5).mean()
    samples = []
    for batch in tqdm(data, desc='quantile_sample_calc'):
      pose = batch[pose_modality]
      pose = pose.reshape(pose.shape[0], 2, -1).transpose(0, 2, 1)
      samples.append(vel(pose, list(range(1, pose.shape[1]))))

    min_sample, max_sample = min(samples), max(samples)
    if kind == 'above':
      v0 = np.quantile(np.array(samples, dtype=np.float), q)
      print('above {} percentile'.format(v0))
    elif kind == 'tail':
      v0 = [np.quantile(np.array(samples, dtype=np.float), q[0]), np.quantile(np.array(samples, dtype=np.float), q[1])]
      print('below {} and above {} percentile'.format(*v0))
    elif kind == 'rebalance':
      v0 = torch.arange(min_sample, max_sample+1e-5, (max_sample - min_sample)/q)
      print('rebalaced data'+(' {}'*len(v0)).format(*v0))
      
    def in_subset(v, v0):
      if kind == 'above':
        return v > v0
      elif kind == 'tail':
        return (v > v0[1]) or (v < v0[0])
      elif kind == 'rebalance':
        starts, ends = v0[:-1], v0[1:]
        interval = ((starts <= v) * (v <= ends))
        if interval.any():
          return interval.int().argmax().item()
        else:
          raise 'incorrect interval'

    if kind in ['tail', 'above']:
      subset_idx = []
    elif kind == 'rebalance':
      subset_idx = [[] for _ in range(len(v0) - 1)]

    for i, batch in tqdm(enumerate(data), desc='quantile subset'):
      pose = batch[pose_modality]
      pose = pose.reshape(pose.shape[0], 2, -1).transpose(0, 2, 1)
      v = vel(pose, list(range(1, pose.shape[1])))
      if kind in ['tail', 'above']:
        if in_subset(v, v0):
          subset_idx.append(i)
      elif kind == 'rebalance':
        subset_idx[in_subset(v, v0)].append(i)

    return subset_idx, kind
      
  def get_train_sampler(self, dataset_train, train_intervals_dict):
    ## Style iterations with AlternateClassSampler
    if self.style_iters > 0 and self.sample_all_styles == 0:
      train_sampler = self.get_alternate_class_sampler(dataset_train, train_intervals_dict, self.style_iters)
    ## Sampler with lesser number of samples for few-shot learning.
    elif self.num_training_sample is not None:
      subset_idx = torch.randperm(len(dataset_train))
      train_sampler = torch.utils.data.SubsetRandomSampler(subset_idx[:self.num_training_sample])
    elif self.quantile_sample is not None:
      subset_idx, kind = self.get_quantile_sample(dataset_train, self.quantile_sample)
      if kind in ['above', 'tail']:
        train_sampler = torch.utils.data.SubsetRandomSampler(subset_idx)
      elif kind in ['rebalance'] and self.quantile_num_training_sample is not None:
        subset_idx = [torch.LongTensor(li) for li in subset_idx]
        train_sampler = BalanceClassSampler(subset_idx, int(self.quantile_num_training_sample)*self.batch_size)
    elif self.weighted:
      train_sampler = torch.utils.data.WeightedRandomSampler([1]*len(dataset_train), self.weighted*self.batch_size)
    elif self.num_training_iters is not None:
      train_sampler = torch.utils.data.RandomSampler(dataset_train, num_samples=self.num_training_iters*self.batch_size, replacement=True)
    else:
      train_sampler = torch.utils.data.RandomSampler(dataset_train)
      #train_sampler = None

    return train_sampler
  
  def close_hdf5_files(self, files):
    for h5 in files:
      h5.close()

  @property
  def shape(self):
    for minidata in self.train.dataset.datasets:
      if len(minidata) > 0:
        break
    shape = {}
    for modality, feats_shape in zip(self.modalities, minidata.shapes):
      start = minidata.idx_start_list_dict[modality][0]
      end = minidata.idx_end_list_dict[modality][0]
      interval = minidata.idx_interval_dict[modality]
      length = len(range(start, end, interval))
      shape.update({modality:[length, feats_shape[-1]]})
    return shape
  
class MiniData(Dataset, HDF5):
  def __init__(self, path2h5, modalities, fs_new, time, modality_classes, window_hop, style=0, repeat_text=1, text_in_modalities=False, filler=0, **kwargs):
    super(MiniData, self).__init__()
    self.path2h5 = path2h5
    self.modalities = modalities
    self.fs_new = fs_new
    self.time = time
    self.modality_classes = modality_classes
    self.window_hop = window_hop
    self.style = style
    self.repeat_text = repeat_text
    self.text_in_modalities = text_in_modalities
    self.filler = filler
    
    ## load modality shapes and maybe data
    self.shapes = []
    self.data = []
    for modality in self.modalities:
      try:
        data, h5 = self.load(self.path2h5, modality)
      except:
        print(self.path2h5, modality)
        sys.exit(1)
        
      self.shapes.append(data.shape)
      self.data.append(data[()])
      h5.close()

    if self.text_in_modalities:
      try:
        self.text_df = pd.read_hdf(self.path2h5, key='text/meta')
      except:
        self.text_df = None
    if self.filler:
      self.stopwords = kwargs['stopwords']
      self.tokenizer = kwargs['tokenizer']

    ## create idx lists
    self.idx_start_list_dict = {}
    self.idx_end_list_dict = {}
    self.idx_interval_dict = {}

    self.update_idx_list(self.time, self.window_hop)
    
  def update_idx_list(self, time, window_hop=0):
    for modality, fs_new, shape in zip(self.modalities, self.fs_new, self.shapes):
      fs = self.modality_classes[modality].fs(modality)
      window = int(time*fs)
      assert window_hop < window, 'hop size {} must be less than window size {}'.format(window_hop, int(time*fs))

      fs_ratio = round(fs/fs_new)
      self.idx_interval_dict[modality] = fs_ratio

      if not window_hop:
        time_splits = np.r_[range(0, shape[0]-window, int(window))]
      else:
        time_splits = np.r_[range(0, shape[0]-window, int(window_hop*fs_ratio))]
      self.idx_start_list_dict[modality] = time_splits[:]
      self.idx_end_list_dict[modality] = time_splits + window

    #len_starts = [len(self.idx_start_list_dict[modality]) for modality in self.idx_start_list_dict]
    #raise len_starts[0] == len_starts[1], 'number of idxes are not consistent in file {}'.format(self.path2h5)
      
  def __len__(self):
    return min([len(self.idx_start_list_dict[modality]) for modality in self.modalities])
    #return len(self.idx_start_list_dict[self.modalities[0]])

  def __getitem__(self, idx):
    item = {}
    ## args.modalities = ['pose/normalize', 'text/w2v']
    for i, modality in enumerate(self.modalities):
      ## read from loaded data
      data = self.data[i]
      
      ## open h5 file
      #data, h5 = self.load(self.path2h5, modality)
      
      start = self.idx_start_list_dict[modality][idx]
      end = self.idx_end_list_dict[modality][idx]
      interval = self.idx_interval_dict[modality]

      item[modality] = data[start:end:interval].astype(np.float64)
      start_time = data[0:start:interval].shape[0] / self.fs_new[-1]

      if 'text' in modality:
        vec = item[modality]
        indices = [0] ## starts in 64 frames
        if self.text_df is None or modality == 'text/tokens': ## to be used with self.repeat_text = 0
          for t in range(1, vec.shape[0]):
            if (vec[t] - vec[indices[-1]]).sum() != 0:
              indices.append(t)
        else:
          text_df_ = self.text_df[(start <= self.text_df['end_frame']) & (end > self.text_df['start_frame'])]
          starts_ = text_df_['start_frame'].values - start
          starts_[0] = 0
          indices = list(starts_.astype(np.int))
        if not self.repeat_text:
          item.update({modality:vec[indices]}) ## if self.repeat_text == 0, update the text modality

        ## add filler masks
        if self.filler:
          filler = np.zeros((len(indices),))
          if self.text_df is None:
            pass ## if text_df is not available, assume no word is filler
          else:
            words = self.text_df[(start <= self.text_df['end_frame']) & (end > self.text_df['start_frame'])].Word.values
            words = [word.lower() for word in words]
            if 'bert' in modality or 'tokens' in modality:
              words = self.tokenizer.tokenize(' '.join(words))

            for i, word in enumerate(words[:len(indices)]):
              if word in self.stopwords:
                filler[i] = 1
          if self.repeat_text:
            filler_ = np.zeros((vec.shape[0], ))
            end_indices = indices[1:] + [vec.shape[0]]
            for i, (st, en) in enumerate(zip(indices, end_indices)):
              filler_[st:en] = filler[i]
            filler = filler_
          item.update({'text/filler':filler})
                

        ## duration of each word
        indices_arr = np.array(indices).astype(np.int)
        length_word = np.zeros_like(indices_arr)
        length_word[:-1] = indices_arr[1:] - indices_arr[:-1]
        duration = (end-start)/interval
        length_word[-1] = duration - indices_arr[-1]
        item.update({'text/token_duration':length_word.astype(np.int)})
      
      ## close h5 file
      #h5.close()
      
    ## start and end times of audio in the interval
    #start_time = self.fs_new[-1] * data[0:start:interval].shape[0]
    duration = item[self.modalities[0]].shape[0]/self.fs_new[-1]
    #duration = ((end-start)/interval)/self.fs_new[-1]
    end_time = start_time + duration
    
    item.update({'meta':{'interval_id':Path(self.path2h5).stem,
                         'start':start_time,
                         'end':end_time,
                         'idx':idx}})

    item['style'] = np.zeros(item[self.modalities[0]].shape[0]) + self.style

    return item
  
  def close_h5_files(self, files):
    for h5 in files:
      h5.close()

class AlternateClassSampler(Sampler):
  def __init__(self, class_count, num_samples, replacement=True):
    self.num_samples_per_class = num_samples//len(class_count)
    self.num_samples = self.num_samples_per_class*len(class_count)
    self.class_count = class_count
    self.starts = [0]
    self.ends = []
    for counts in self.class_count:
      self.starts.append(self.starts[-1]+counts)
      self.ends.append(self.starts[-1])
    self.starts = self.starts[:-1]

  def __iter__(self):
    return iter(torch.stack([torch.randint(start, end, size=(self.num_samples_per_class, )) for start, end in zip(self.starts, self.ends)], dim=1).view(-1).tolist())
  
  def __len__(self):
    return self.num_samples

class BalanceClassSampler(Sampler):
  def __init__(self, classes, num_samples, replacement=True):
    self.classes = classes
    self.update_classes()
    self.num_samples_per_class = num_samples//len(self.classes)
    self.num_samples = self.num_samples_per_class*len(self.classes)

  def update_classes(self):
    cl_list = []
    for cl in self.classes:
      if cl.shape[0] > 0:
        cl_list.append(cl)
    self.classes = cl_list
    
  def __iter__(self):
    return iter(torch.stack([class_idx[torch.randint(0, len(class_idx), size=(self.num_samples_per_class,))] for class_idx in self.classes], dim=1).view(-1).tolist())
  
  def __len__(self):
    return self.num_samples

class ConcatDatasetIndex(ConcatDataset):
  def __init__(self, datasets):
    super().__init__(datasets)

  def __getitem__(self, idx):
    if idx < 0:
      if -idx > len(self):
        raise ValueError("absolute value of index should not exceed dataset length")
      idx = len(self) + idx
    dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
    if dataset_idx == 0:
      sample_idx = idx
    else:
      sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
    batch = self.datasets[dataset_idx][sample_idx]
    if isinstance(batch, dict):
      batch.update({'idx':idx})
    return batch

    
def unittest(args, exp_num):
  path2data = args.path2data
  speaker = args.speaker
  modalities = args.modalities
  fs_new = args.fs_new
  time = args.time
  split = args.split
  batch_size = args.batch_size
  shuffle = args.shuffle

  data = Data(path2data=path2data,
              speaker=speaker,
              modalities=modalities,
              fs_new=fs_new,
              time=time,
              split=split,
              batch_size=batch_size,
              shuffle=shuffle)

  print('Speaker: {}'.format(speaker))
  for batch in tqdm(data.train):
    continue
  sizes = {modality:batch[modality].shape for modality in modalities}
  print('train')
  print(sizes)

  for batch in tqdm(data.dev):
    continue
  sizes = {modality:batch[modality].shape for modality in modalities}
  print('dev')
  print(sizes)

  for batch in tqdm(data.test):
    continue
  sizes = {modality:batch[modality].shape for modality in modalities}
  print('test')
  print(sizes)
  
if __name__ == '__main__':
  argparseNloop(unittest)
