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

from common import Modality, MissingData

class Audio(Modality):
  def __init__(self, path2data='../dataset/groot/data',
               path2outdata='../dataset/groot/data',
               speaker='all',
               preprocess_methods=['log_mel_512']):
    super(Audio, self).__init__(path2data=path2data)
    self.path2data = path2data
    self.df = pd.read_csv(Path(self.path2data)/'cmu_intervals_df.csv', dtype=object)
    self.df.loc[:, 'delta_time'] = self.df['delta_time'].apply(float)
    self.df.loc[:, 'interval_id'] = self.df['interval_id'].apply(str)
    
    self.path2outdata = path2outdata
    self.speaker = speaker
    self.preprocess_methods = preprocess_methods

    self.missing = MissingData(self.path2data)
    
  def log_mel_512(self, y, sr, eps=1e-10):
    spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512)
    mask = (spec == 0).astype(np.float)
    spec = mask * eps + (1-mask) * spec
    return np.log(spec).transpose(1,0)

  def log_mel_400(self, y, sr, eps=1e-6):
    y = librosa.core.resample(y, orig_sr=sr, target_sr=16000) ## resampling to 16k Hz
    #pdb.set_trace()
    sr = 16000
    n_fft = 512
    hop_length = 160
    win_length = 400
    S = librosa.core.stft(y=y.reshape((-1)),
                          n_fft=n_fft,
                          hop_length=hop_length,
                          win_length=win_length,
                          center=False)
                                   
    S = np.abs(S)
    spec = librosa.feature.melspectrogram(S=S, 
                                          sr=sr, 
                                          n_fft=n_fft, 
                                          hop_length=hop_length, 
                                          power=1,
                                          n_mels=64,
                                          fmin=125.0,
                                          fmax=7500.0,
                                          norm=None)    
    mask = (spec == 0).astype(np.float)
    spec = mask * eps + (1-mask) * spec
    return np.log(spec).transpose(1,0)

  def silence(self, y, sr, eps=1e-6):
    vad = webrtcvad.Vad(3)
    y = librosa.core.resample(y, orig_sr=sr, target_sr=16000) ## resampling to 16k Hz
    #pdb.set_trace()
    fs_old = 16000
    fs_new = 15
    ranges = np.arange(0, y.shape[0], fs_old/fs_new)
    starts = ranges[0:-1]
    ends = ranges[1:]

    is_speeches = []
    for start, end in zip(starts, ends):
      Ranges = np.arange(start, end, fs_old/100)
      is_speech = []
      for s, e, in zip(Ranges[:-1], Ranges[1:]):
        try:
          is_speech.append(vad.is_speech(y[int(s):int(e)].tobytes(), fs_old))
        except:
          pdb.set_trace()
      is_speeches.append(int(np.array(is_speech, dtype=np.int).mean() <= 0.5))
      is_speeches.append(0)
    return np.array(is_speeches, dtype=np.int)

  @property
  def fs_map(self):
    return {
      'log_mel_512': int(45.6*1000/512), #int(44.1*1000/512) #112 #round(22.5*1000/512)
      'log_mel_400': int(16.52 *1000/160),
      'silence': 15
      }
  
  def fs(self, modality):
    modality = modality.split('/')[-1]
    return self.fs_map[modality]

  @property
  def h5_key(self):
    return 'audio'
