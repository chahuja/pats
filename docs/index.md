# PATS Dataset (**P**ose, **A**udio, **T**ranscript, **S**tyle) <img src="https://user-images.githubusercontent.com/43928520/90432137-1cbcaf80-e098-11ea-8491-0f7c92da4b29.png" width="100" height="100">

* Contains transcribed **Pose** data with aligned **Audio** and **Transcriptions**
    - 25 Speakers with different **Styles** 
    - Includes 10 speakers from [Ginosar, et al. (CVPR 2019)](https://people.eecs.berkeley.edu/~shiry/projects/speech2gesture/index.html) )
    - 15 talk show hosts, 5 lecturers, 3 YouTubers, and 2 televangelists
* 251 hours of data 
    - Mean: 10.7s per interval
    - Standard Deviation: 13.5s per segment


<img src="https://user-images.githubusercontent.com/43928520/90454983-c022ba00-e0c2-11ea-991e-36bd5cb3b38b.png" width="1000">

## Dataset Features

| Features | Available Representations | Description |
| :--- | :---: | ---: |
| Audio | Log-mel Spectrograms | Audio directly scraped from Youtube |
| Language | BERT, Word2Vec | Transcript derived from [Google ASR](https://cloud.google.com/speech-to-text), WER = 0.29, Bert uses [HuggingFace](https://huggingface.co/transformers/model_doc/bert.html) |
| Pose | OpenPose Skeletal Keypoints | Derived from [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) |


## Link to data:
Data can be downloaded here : LINK HERE\
We've also included a csv file, in which you can see the video source and the specific section we took the intervals from.


## Requirements:

## Tasks



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
