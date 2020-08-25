
# PATS Dataset
PATS was collected to study correlation of co-speech gestures with audio and text signals. The dataset consists of a diverse and large amount of aligned pose, audio and transcripts. With this dataset, we hope to provide a benchmark which would help develop technologies for virtual agents which generate natural and relevant gestures.
For a complete overview check the following [link](http://chahuja.com/pats). 

<center>
<img src="https://user-images.githubusercontent.com/43928520/90454983-c022ba00-e0c2-11ea-991e-36bd5cb3b38b.png" width="500px">
</center>

## Relevant Paper(s)
- Ahuja, Chaitanya, et al. "Style Transfer for Co-Speech Gesture Animation: A Multi-Speaker Conditional-Mixture Approach." ECCV 2020 - [[website](http://chahuja.com/mix-stage)][[code](http://github.com/chahuja/mix-stage)]

# Structure of the dataset

```sh
pats/data
  - cmu_intervals_df.csv
  - missing_intervals.h5
  - processed
      - oliver # speakers
          - XXXX.h5
          - YYYY.h5
          ...
      - jon
      ...
      ...
      - bee
      - noah
  - raw
      - oliver_cropped
          - xxx.mp3
```

The dataset consists of:

- `cmu_intervals_df.csv`: list of all intervals and the relevant meta information (Similar to [Ginosar et. al. 2019](https://github.com/amirbar/speech2gesture/blob/master/data/dataset.md))
- `missing_intervals.h5`: list of all intervals that have incomplete set of features. For the sake of uniformity they are excluded from the benchmark tests.
- `proceesed`: h5 files containing processed features for pose, audio and transcripts for all speakers
- `raw`: mp3 audio files corresponding for each interval which is useful during rendering.

## Processed Features
Heirarchy of the features in h5 files explained. To access a festure, both parent and child combine to give a key. For example, `pose/data`, `audio/log_mel_400` and so on.
- `pose/` 
    - `data`: XY coordinates of upper body pose relative to the neck joint. Joint order and parents can be found [here](https://github.com/chahuja/pats/blob/master/data/skeleton.py#L40)
    - `normalize`: same as data but the size of the body is normalized across speakers. In other words, each speaker is scaled to have the same shoulder length. This is especially useful in style transfer experiments where we would like the style of gestures to be independent of the size of the speaker.
    - `confidence`: confidence scores provided by openpose with 1 being most confident and 0 being least confident.
- `audio/`
    - `log_mel_400`:  Log mel Spectrograms extracted with the function [here](https://github.com/chahuja/pats/blob/master/data/audio.py#L38)
    - `log_mel_512`: Log mel Spectrograms extracted with the function [here](https://github.com/chahuja/pats/blob/master/data/audio.py#L32) 
    - `silence`: Using [VAD](https://github.com/chahuja/pats/blob/master/data/audio.py#L65) we estimate which segments have voice of the speaker and which just have noise.
- `text/`
    - `bert`: fixed pre-trained bert embeddings of size 768
    - `tokens`: tokens extracted using BertTokenizer of [HuggingFace](https://huggingface.co)
    - `w2v`: Word2Vec features of size 300
    - `meta`: Pandas Dataframe with words, start_frame and end_frame

## Raw Features
We provide links to original youtube videos in `cmu_intervals_df.csv` to help download the relevant audio files. Rendering the generated animations with audio would require the raw audio and would be useful for user-studies.

# Dataset Download
To download **processed** features of the dataset visit [here](http://chahuja.com/pats/download.html).

To download **raw** features of the dataset run,
```sh
python youtube2croppedaudio/youtube2audio.py \
-base_path pats/data/ \ # Path to dataset folder 
-speaker bee \ # Speaker Name (Optional). Downloads all speakers if not specified
-interval_path cmu_intervals_df.csv
```
As raw audio files are downloaded from video streaming websites such as YouTube, some of them may not be available at the time of download. For the purposes of consistent benchmarking **processed** features should be used.

# Data Loader
As a part of this dataset we provide a DataLoader in [PyTorch](https://pytorch.org) to jumpstart your research. This DataLoader samples batches of aligned processed features of Pose, Audio and Transcripts for one or many speakers in a dictionary format. We describe the various [arguments](#arguments-of-class-data) of the class [`Data`](https://github.com/chahuja/pats/blob/master/data/dataUtils.py#L51) which generates the DataLoaders.

DataLoader Examples: [Ipython Notebook](dataloader_tutorial.ipynb)

## Requirements
* pycasper

```sh
mkdir ../pycasper
git clone https://github.com/chahuja/pycasper ../pycasper
ln -s ../pycasper/pycasper .
```

* Create an [anaconda](https://www.anaconda.com/) or a virtual enviroment and activate it

```sh
pip install -r requirements.txt
```

## Arguments of class `Data`
There are way too many arguments (#research) for `Data`. For most cases you might not even need most of them and can leave them as default values. We divide the arguments into **Essential**, **DataLoader Arguments**, **Modality Arguments**, **Sampler Arguments** and **Others**.
### Essential
- `path2data (str)`: path to processed data e.g. "pats/data/processed"
- `speaker (str or list)`: one or more speaker names. Find list of speakers [here](https://github.com/chahuja/pats/blob/master/data/common.py#L152).
- `modalities (list)`: list of processed features to be loaded. Default- ['pose/data', 'audio/log_mel_512']. Find list of all processed features [here](https://github.com/chahuja/pats#processed-features).
- `fs_new (list)`: list of frame rates for each modality in modalities. Default- [15, 15]. Length of fs_new == Length of modalities. 
- `time (float)`: length of window for each sample in seconds. Default- 4.3. The default value is recommended. It results in 64 frames of audio and pose when fs_new is 15.
- `split (tuple or None)`: train, dev and test split as fractions. Default- None. Using None would use pre-defined splits in cmu_intervals_df.csv. Example use case of a tuple, (0.7, 0.1) represents the ratios of train and dev, hence test split is 0.2.
- `window_hop (int)`: number of frames a window hops in an interval to contruct samples. Default- 0. Using 0 implies non-overlapping windows. For `window_hop` > 0, samples are created with the following formula `[sample[i:i+int(time*fs_new[0])] for i in range(0, len(sample), window_hop)]`

### DataLoader Arguments
- `batch_size (int)`: Size of batch. Default- 100.
- `shuffle (bool)`: Shuffle samples after each epoch. Default- True
- `num_workers (int)`: Number of workers to load the data. Defaut- 0

### Text Arguments
- `filler (int)`: Get "text/filler" as a feature in the sampled batch. This feature is a tensor of shape 'batch x time', where each element represents if the spoken work was a filler word or not. The list of filler words is the same as nltk's stopword list for english. Default- 0. Use 1 to get the "text/filler" feature.
- `repeat_text (int)`: If 1, the feature of each word token is repeated to match the length of its duration. For example if a word is spoken for 10 frames of the pose and/or audio sequence, it is stacked 10 times. Hence the time dimension of pose audio and transcripts are the same. If 0, words tokens are not repeated. As each sample could have different number of words, the shorter sequences are padded with zeros. Extra features "text/token_duration" and "text/token_count" are also part of the sample which represent the duration of each token in frames and number of tokens in each sequence respectively.

### Sampler Arguments (Mutually exclusive unless specified)
- `style_iters (int)`: If value > 0, [`AlternateClassSampler`](https://github.com/chahuja/pats/blob/master/data/dataUtils.py#L618) is used as the sampler argument while building the train dataloader. This sampler is useful if two or more speakers are trained together. This sampler ensures that each mini-batch has equal number of samples from each speaker. Value refers to the number of iterations in each epoch. Default- 0.
- `sample_all_styles (int)`: Can only be used with argument `style_iters` If value > 0, randomly selects value number of samples from each speaker to load. This is especially useful for performing inference in style transfer experiments, when the number of permutations of style transfer increases exponentially with the number of speakers. This argument puts an upper bound on the number of samples for each speaker, hence limiting the time to generate gestures for a limited number of inputs. Default- 0.
- `num_training_sample (int or None)`: if value > 0, chooses a random subset of unique samples with cardinality of the set == value as the new training set. if value is None, all samples are considered for training. Default- None.
- `quantile_sample (float or int or None)`: Default- None.
- `quantile_num_training_sample (int or None)`: Default- None.
- `weighted (int)`: If value > 0, `torch.utils.data.WeightedRandomSampler` as the sampler argument while building the train dataloader. The weights are set to 1 for each sample. While, this is equivalent to a uniform sampler, this provides a possibility of being able to change the weights for each sample while training. Default- 0.

### Others
- `load_data (bool)`: If True, loads the hdf5 files in RAM. If False, files are not loaded and the dataloaders will not work as intended. Useful for quick debugging.
- `num_training_iters (int or None)`: If value > 0, changes the training sampler to sample with replacement and value is the number of iterations per epoch. If value is None, the sampler samples without replacement and the number of iterations are inferred based on the size of the dataset. Default- None.

## Render
Check the repository for [Mix-StAGE](https://github.com/chahuja/mix-stage) for rendering scripts.

# Creating your own dataloader
In case you prefer to create your own dataloaders, we would recommend checking out the [structure of the h5 files](#processed-features) and the last sections of the [Ipython Notebook](dataloader_tutorial.ipynb). We have a class [`HDF5`](data/common.py#L16) with many staticmethods which might be useful to load HDF5 files.

# Issues
All research has a tag of work in progress. If you find any issues with this code, feel free to raise issues or pull requests (even better) and I will get to it as soon as humanly possible.
