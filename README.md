**Fruits Classification Using CNN**

In this project, I developed a Convolutional Neural Network (CNN) model to classify various types of fruits using the Fruits 360 dataset from Kaggle. This dataset contains 5605 images of fruits categorized into 21 different classes. The classes include different varieties of apples, cabbage, carrot, cucumber, eggplant, and zucchini, among others. The goal was to build a robust and accurate model that can effectively classify these fruit images.

**Dataset and Preprocessing**

The Fruits 360 dataset is a well-structured dataset specifically designed for image classification tasks. It consists of high-quality images of fruits with each class having a significant number of images to ensure diversity and reduce overfitting. For this project, the images were preprocessed by resizing them to a uniform size and normalizing the pixel values to a range between 0 and 1. This preprocessing step is crucial as it ensures that the input data is in a consistent format, which helps in improving the performance of the CNN model.

**Model Architecture**

The model used in this project is a Sequential CNN Model. The architecture includes four Conv2D (convolutional) layers followed by MaxPool2D (max pooling) layers. The Conv2D layers are responsible for detecting various features in the images, such as edges, textures, and shapes. These layers use different filters to perform convolutions on the input image, capturing important features that are essential for classification. The MaxPool2D layers, on the other hand, are used to reduce the spatial dimensions of the feature maps, which helps in reducing the computational complexity and preventing overfitting.

After the convolutional and pooling layers, the feature maps are flattened into a single vector, which is then fed into a Dense (fully connected) layer. This Dense layer is responsible for performing the final classification. The model uses the ReLU (Rectified Linear Unit) activation function in the hidden layers to introduce non-linearity, and the Softmax activation function in the output layer to generate probabilities for each class.

**Training and Evaluation**

The model was trained using the training subset of the Fruits 360 dataset. During training, the model learns the optimal weights and biases for the filters and neurons through backpropagation and gradient descent. The training process involves minimizing the categorical cross-entropy loss, which measures the difference between the predicted probabilities and the actual class labels. The Adam optimizer was used to update the weights, as it is well-suited for handling large datasets and complex models.

The model's performance was evaluated on a separate test dataset, which contains images that the model has not seen during training. The accuracy, precision, recall, and F1 score were calculated to assess the model's performance. The CNN model achieved high accuracy, demonstrating its effectiveness in classifying different types of fruits.

**Results and Predictions**

After training, the model was tested on new, unseen images to predict their classes. The predictions were highly accurate, with the model correctly identifying the fruit types in most cases. The results were visualized using confusion matrices and classification reports, which provided a detailed analysis of the model's performance across different classes. This step is crucial as it helps in identifying any potential areas of improvement, such as classes where the model might be underperforming.

**Conclusion**

The Fruits Classification Using CNN project successfully demonstrated the power of convolutional neural networks in image classification tasks. By leveraging the Fruits 360 dataset, the project was able to build a robust and accurate model that can classify various fruit types with high precision. This project not only highlights the practical applications of deep learning in agriculture and food industry but also showcases the importance of proper data preprocessing, model architecture design, and thorough evaluation in developing effective machine learning models.








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
