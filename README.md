# Fruits_Classification_Using_CNN
Convolutional Neural Network

Used the following Fruits 360 dataset from Kaggle:

https://www.kaggle.com/moltean/fruits

# TRAINING:

Trained the CNN for 5605 images belonging to 21 different classes. Following are the classes for which our CNN classifier will be trained for with their class indices

{'Apple rotten': 0,
 'apple6': 1,
 'apple_braeburn': 2,
 'apple_crimson_snow': 3,
 'apple_golden_1': 4,
 'apple_golden_2': 5,
 'apple_golden_3': 6,
 'apple_granny_smith': 7,
 'apple_hit': 8,
 'apple_pink_lady': 9,
 'apple_red_1': 10,
 'apple_red_2': 11,
 'apple_red_3': 12,
 'apple_red_delicios': 13,
 'apple_red_yellow': 14,
 'cabbage_white': 15,
 'carrot': 16,
 'cucumber': 17,
 'eggplant_violet': 18,
 'zucchini': 19,
 'zucchini_dark': 20}
 
 # MODEL SUMMARY:
 
 The model which is used is a Sequential CNN Model with 4 Recurrent conv2D and MaxPool2D layers, with Flattening and Dense layer at the end. 
 
 The model summary is as follows:
 
 ![capture_1](https://user-images.githubusercontent.com/58310295/134747304-6622db7e-7be3-495a-8d8a-63358ecd12a3.JPG)

# PREDICTION:

After training we predict on test dataset. The output result we get is of the following form:

![Capture_2](https://user-images.githubusercontent.com/58310295/134747595-0693f378-0310-4dfa-ad17-7a83a3113c2f.JPG)

![Capture_3](https://user-images.githubusercontent.com/58310295/134747646-0e78d97e-08d4-4a94-b1b2-509205457094.JPG)
