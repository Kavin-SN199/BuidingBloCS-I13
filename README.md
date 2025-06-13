
#  ASL Alphabet Interpreter  
Recognize American Sign Language letters using deep learning!

This project trains a Convolutional Neural Network (CNN) to classify hand gestures from grayscale images into 28 classes — the English alphabet (A–Z), plus `space` and `nothing`. It supports training and webcam inference.

---

##  Features

-  29-class classification: A–Z, space, nothing  
-  Trained on 87,000+ preprocessed hand gesture images  
-  Run predictions in real time using your webcam  





Required packages:

tensorflow
pandas
numpy
scikit-learn
opencv-python
matplotlib
joblib



##  Dataset

The dataset was sourced from [Kaggle: ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) and converted into a single `.csv` with pixel-normalized grayscale values for training.

### Label Map:


A → 0, B → 1, ..., Z → 25, space → 26, nothing → 27

## How to start the Translation.
1. Download the asl_cnn_model.h5
2. Copy the repository into Visual Studio Code
3. Tweak the code in the DeployModel.py such that it matches the path in your computer
4. Run the (DeployModeel.py) Python script in Visual Studio Code (This is for easier access to the computer's webcam and for accessing the downloaded model)
5. Replace the file path with the file path of the downloaded model.
6. Run and have fun


## Tips to improve recognition
 Tip                          Why It Matters                                        

 Good lighting      ->     Helps to detect the hand more accurately.      
Solid background     ->    Avoid noisy backgrounds for better hand segmentation. 
Show one hand      ->       The code is set to detect only 1 hand.                
Hold the gesture still  ->  Prediction is per frame; stability helps confidence.  
Hand centered in camera  => Ensures the whole hand is captured.         


##Slides
Link to the slides->https://www.canva.com/design/DAGqOsR60r8/cNsrxVEgU-G7HVasuiyNpg/edit

##  Acknowledgments

* Dataset by [grassknoted on Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
* Help of A.I. tools for brainstorming
* A bit of help of A.I. tools to understand on how to convert the dataset to a csv file







