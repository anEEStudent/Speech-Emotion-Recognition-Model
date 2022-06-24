## Description
Our team wrote this script for a ML hackathon (Brainhack by DSTA) which involved creating a Speech Recognition Model which could classify human voices by emotion. The 5 emotions for classification were angry, fear, happy, neutral, and sad. The dataset provided was mostly open-source data with the test data being Singaporean voices recorded by the organisers of the event. Our model achieved roughly ~55% accuracy during the evaluation phase, which placed us in second place for this section of the even along with our [CV model](www.google.com). 

## Overview
There are two scripts prepared: the script we used to train the model and the script for inferencing only. The model script can be used as it is if you wish to train a new model. Only the dependencies need to be installed.

## Description of model
TODO


## Acknowledgements:
The architecture of our model was inspired by the following open-source model:
https://github.com/maelfabien/Multimodal-Emotion-Recognition/blob/master/README.md#v-how-to-use-it-

We used some additional data from the RAVDESS open-source dataset on top of the data provided:
https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio
"The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)" by Livingstone & Russo is licensed under CC BY-NA-SC 4.0.
Specifically, we added in actors 1-10 for all emotions except for neutral where we added in actor 3 twice and left out actor 1.

Team members: @anEEStudent, @dominic-soh, @GordonShinozaki

