# PATS (Pose, Audio, Transcript, Style) Dataset

The PATS dataset contains aligned transcribed language, audio, gesture data for 25 speakers (including 10 speakers from [Ginosar, et al.](https://people.eecs.berkeley.edu/~shiry/projects/speech2gesture/index.html) *) to offer a total of 251 hours of data, with a mean of 10.7 seconds and a standard deviation of 13.5 seconds per interval. The demographics of the speakers include 15 talk show hosts, 5 lecturers, 3 YouTubers, and 2 televangelists.

The dataset includes the following 3 features:

Aligned Transcriptions: As manual transcriptions are often not aligned and not readily available, we use Google Automatic Speech Recognition to collect subtitles and aligned timings of each spoken word.he average Word Error Rate of the transcriptions, calculated on the set of available transcriptions (i.e.  manually inputted Youtube subtitles), using the Fisher-Wagner algorithm is 0.29.

Pose: Each speaker’s pose is represented via skeletal keypoints collected via OpenPose. It consists of of 52 coordinates of an individual’s major joints for each frame at 15 framesper second, which we rescale by holding the length of each individual’s shoulder constant.

Audio: We represent audio features as spectrograms, which is a rich input represen-tation shown to be useful for gesture generation.

The dataset is used in:
[Style Transfer for Co-Speech Gesture Animation: A Multi-Speaker Conditional-Mixture Approach](https://arxiv.org/abs/2007.12553) *

## Table of Contents
1. Link to data
2. Code
3. Examples
4. Reference


### Link to data:
Data can be downloaded here : LINK HERE
We've also included a csv file, in which you can see the video source and the specific section we took the intervals from.


### Code

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```


### Reference
If you found this code useful, please cite the following paper:
