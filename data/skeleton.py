import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime

from common import Modality, MissingData

from pathlib import Path
import numpy as np
import pandas as pd
import pdb
from tqdm import tqdm
import yaml
import warnings

from pycasper.pathUtils import replace_Nth_parent

class Skeleton2D(Modality):
  def __init__(self, path2data='../dataset/groot/data/speech2gesture_data',
               path2outdata='../dataset/groot/data',
               speaker='all',
               preprocess_methods=['data']):
    super(Skeleton2D, self).__init__(path2data=path2data)
    self.path2data = path2data
    self.df = pd.read_csv(Path(self.path2data)/'cmu_intervals_df.csv', dtype=object)
    self.df.loc[:, 'delta_time'] = self.df['delta_time'].apply(float)
    self.df.loc[:, 'interval_id'] = self.df['interval_id'].apply(str)
    
    self.path2outdata = path2outdata
    self.speaker = speaker
    self.preprocess_methods = preprocess_methods
    
    self.missing = MissingData(self.path2outdata)



  @property
  def parents(self):
    return [-1,
            0, 1, 2,
            0, 4, 5,
            0, 7, 7,
            6,
            10, 11, 12, 13,
            10, 15, 16, 17,
            10, 19, 20, 21,
            10, 23, 24, 25,
            10, 27, 28, 29,
            3,
            31, 32, 33, 34,
            31, 36, 37, 38,
            31, 40, 41, 42,
            31, 44, 45, 46,
            31, 48, 49, 50]

  @property
  def joint_subset(self):
    ## choose only the relevant skeleton key-points (removed nose and eyes)
    return np.r_[range(7), range(10, len(self.parents))]

  @property
  def root(self):
    return 0

  @property
  def joint_names(self):
    return ['Neck',
            'RShoulder', 'RElbow', 'RWrist',
            'LShoulder', 'LElbow', 'LWrist',
            'Nose', 'REye', 'LEye',
            'LHandRoot',
            'LHandThumb1', 'LHandThumb2', 'LHandThumb3', 'LHandThumb4',
            'LHandIndex1', 'LHandIndex2', 'LHandIndex3', 'LHandIndex4',
            'LHandMiddle1', 'LHandMiddle2', 'LHandMiddle3', 'LHandMiddle4',
            'LHandRing1', 'LHandRing2', 'LHandRing3', 'LHandRing4',
            'LHandLittle1', 'LHandLittle2', 'LHandLittle3', 'LHandLittle4',
            'RHandRoot',
            'RHandThumb1', 'RHandThumb2', 'RHandThumb3', 'RHandThumb4',
            'RHandIndex1', 'RHandIndex2', 'RHandIndex3', 'RHandIndex4',
            'RHandMiddle1', 'RHandMiddle2', 'RHandMiddle3', 'RHandMiddle4',
            'RHandRing1', 'RHandRing2', 'RHandRing3', 'RHandRing4',
            'RHandLittle1', 'RHandLittle2', 'RHandLittle3', 'RHandLittle4'
    ]

  def fs(self, modality):
    return 15

  @property
  def h5_key(self):
    return 'pose'
