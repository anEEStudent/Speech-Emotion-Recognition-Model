## Description
Our team wrote this script for a ML hackathon ([Brainhack by DSTA](https://www.dsta.gov.sg/latest-news/spotlight/altering-reality-with-brainhack)). The ML component of the hackathon involved creating a Speech Recognition Model which classifies human voices by emotion and a [CV model](https://github.com/GordonShinozaki/yolov5Brainhack) which detects fallen and standing people. The 5 emotions for classification were angry, fear, happy, neutral, and sad. The dataset we trained on was provided by the organisers; consisting of mostly open-source data combined with some voices recorded by the organisers themselves. The test data our model was evaluted on were Singaporean voices recorded by the organisers of the event. Our model achieved roughly ~55% accuracy during the evaluation phase, which placed us in second place for this section of the event along with our [CV model](https://github.com/GordonShinozaki/yolov5Brainhack). 

## Overview
There are two scripts prepared: the script we used to train the model and the script for inferencing only. The model script can be used as it is if you wish to train a new model. However, you will need to use your own training data since the training data we used was provided for the purposes of the competition only. 

## Description of model
The architecture of our model was inspired by the following open-source model:
https://github.com/maelfabien/Multimodal-Emotion-Recognition/blob/master/README.md#v-how-to-use-it-


## Acknowledgements:
The architecture of our model was inspired by the following open-source model:
https://github.com/maelfabien/Multimodal-Emotion-Recognition/blob/master/README.md#v-how-to-use-it-

We used some additional data from the RAVDESS open-source dataset on top of the data provided by the organisers:
https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio
"The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)" by Livingstone & Russo is licensed under CC BY-NA-SC 4.0.

Team members: @anEEStudent, [@dominic-soh](https://github.com/dominic-soh), [@GordonShinozaki](https://github.com/GordonShinozaki)

