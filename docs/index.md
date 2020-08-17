

# PATS (Pose, Audio, Transcript, Style) Dataset

<img src="https://user-images.githubusercontent.com/43928520/90432137-1cbcaf80-e098-11ea-8491-0f7c92da4b29.png" width="100" height="100">

## Overview of PATS
* Contains transcribed language, audio, pose data (3 features)
* 251 hours of data (Mean: 10.7s, Standard Deviation: 13.5s)
* 425 Speakers (including 10 speakers from [Ginosar, et al.](https://people.eecs.berkeley.edu/~shiry/projects/speech2gesture/index.html) )
* 15 talk show hosts, 5 lecturers, 3 YouTubers, and 2 televangelists.
* Includes various representations of features/


## Dataset Features
| Features | Available Representations | Description |
| :---: | :---: | :---: |
| Audio | Log-mel Spectrograms | Audio directly scraped from Youtube |
| Language | BERT, Word2Vec | Transcript derived from Google ASR, WER = 0.29, Bert uses [HuggingFace](https://huggingface.co/transformers/model_doc/bert.html) |
| Gestures | OpenPose Skeletal Keypoints | [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) |



## Table of Contents
1. [Link to data](#Link-to-data)
2. [Requirements](#Requirements)
3. [Example Script](#Examples)
4. [Reference](#Reference)


## Link to data:
Data can be downloaded here : LINK HERE\
We've also included a csv file, in which you can see the video source and the specific section we took the intervals from.


## Requirements:

## Examples

Arguments:
- path2data (str): path to dataset.
- speaker (str): speaker name.
- modalities (list of str): list of modalities to wrap in the dataloader. These modalities are keys of the hdf5 files which were preprocessed earlier (default: ``['pose/data', 'audio/log_mel']``)
- fs_new (list, optional): new frequency of modalities, to which the data is up/downsampled to. (default: ``[15, 15]``).
- time (float, optional): time snippet length in seconds. (default: ``4.3``).
- split (tuple or None, optional): split fraction of train and dev sets. Must add up to less than 1. If ``None``, use ``dataset`` columns in the master dataframe (loaded in self.df) to decide train, dev and test split. (default: ``None``).
- batch_size (int, optional): batch size of the dataloader. (default: ``100``).
- shuffle (boolean, optional): set to ``True`` to have the data reshuffled at every epoch (default: ``False``).
- num_workers (int, optional): set to values >0 to have more workers to load the data. argument for torch.utils.data.DataLoader. (default: ``15``). 
- window_hop
-load_data
- style_iters
- num_training_sample
- nample_all_styles
- repeat_text
- quantile_sample
- quantile_num_training_sample
- weighted
- filler

```markdown
from data.dataUtils import Data 
data = Data('../path/to/data/', 'oliver', ['pose/data', 'audio/log_mel_512', 'text/bert']

for batch in data.train:
    break
    print(batch).
```

### Reference
If you found this dataset helpful, please cite the following paper:

```
@misc{ahuja2020style,
    title={Style Transfer for Co-Speech Gesture Animation: A Multi-Speaker Conditional-Mixture Approach},
    author={Chaitanya Ahuja and Dong Won Lee and Yukiko I. Nakano and Louis-Philippe Morency},
    year={2020},
    eprint={2007.12553},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
