## Description
Our team wrote this script for a ML hackathon which involved creating a Speech Recognition Model which could classify human voices by emotion. The dataset provided was mostly open-source data with the test data being Singaporean voices recorded by the organisers of the event. Our model achieved roughly ~55% accuracy during the evaluation phase.

## Overview
The script can be run as it is to obtain a model and inference. Only the dependencies need to be installed.
All preprocessing work is done in the main script, only the directories in lines 85-90 need to be changed.

Preprocessing steps:
1) Resample audio to 16000 sr 
2) Change all audio length to 48000; pad audio shorter than 48000, cut audio more than 48000
3) Obtain the mel spectrogram of all audio clips


## Acknowledgements:
The architecture of our model was inspired by the following open-source model:
https://github.com/maelfabien/Multimodal-Emotion-Recognition/blob/master/README.md#v-how-to-use-it-

We used some additional data from the RAVDESS open-source dataset on top of the data provided:
https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio
"The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)" by Livingstone & Russo is licensed under CC BY-NA-SC 4.0.
Specifically, we added in actors 1-10 for all emotions except for neutral where we added in actor 3 twice and left out actor 1.


