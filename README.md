﻿# ASLTrainingFiles
These are all of the training files that I wrote for the SignSense project

# Methodology
- Keypoints are extracted via webcam capture through mediapipe
- Those mediapipe key points are stored in an array 30 times (once per frame)
- A three-layer LSTM is trained off of the data from collection
- That model is then loaded by the gesture prediction

# What Makes Mine Different
- This model is trained with a fully custom LSTM that can easily be upscaled
- Unlike others, this is trained off of vector translations and relative positions so it is able to detect from a sequence of frames
- Reduction of unnecessary points added by mediapipe leading to more accurate predictions off of way less data
- Extremely simple and fast to add another gesture and re-train.

# How To Use
- First run the data_collection.py file. Click "s", then record your gesture
- After you've recorded all of your gestures run the train_model.py file and it will automatically train the LSTM model with those key points
- Run the predict_gesture.py and thats it!

# Media
![training](https://github.com/user-attachments/assets/00ae7323-2e3c-4afe-8b8f-e28a00940453)
![prediction](https://github.com/user-attachments/assets/36d992be-2bdd-47ab-bc58-0ffc790a4aaa)
