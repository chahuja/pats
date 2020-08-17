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
2. CSV file with specific intervals and source
3. Examples

test

You can use the [editor on GitHub](https://github.com/chahuja/PATS/edit/master/docs/index.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

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

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/chahuja/PATS/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.


### Reference
If you found this code useful, please cite the following paper:

```
@InProceedings{ginosar2019gestures,
  author={S. Ginosar and A. Bar and G. Kohavi and C. Chan and A. Owens and J. Malik},
  title = {Learning Individual Styles of Conversational Gesture},
  booktitle = {Computer Vision and Pattern Recognition (CVPR)}
  publisher = {IEEE},
  year={2019},
  month=jun
}
```
