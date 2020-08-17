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

## Example

```markdown

from data import Data

data = Data('../path/to/data/', 'lec_cosmic', ['pose/data, audio/logtext/bert'])

```


### Examples

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
