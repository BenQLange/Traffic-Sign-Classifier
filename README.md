# Traffic Sign Classifier

## Overview

This project is part of my Self-Driving course at Udacity. It uses deep neural networks and convolutional neural networks to classify traffic signs. The model was trained using German traffic signs dataset (available here: http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

## Code

Dataset with German Traffic Sign was divided into three datasets: training dataset (train.p), testing dataset (test.p) and validation dataset (valid.p). Size of those files was large so i didn't uploaded by but they an be created using dataset available at the link above.  

1. Preprocessing

Before training the model, figures have been pre-processed. This included converting the pictures to grayscale (grayscale function) using cv2.cvtColor and subsequently, normalizing them in a normalize_grayscale function. The normalizing and grayscale makes a problem easier to optimize by making it well conditioned and getting rid of unnecessary data like colours (1 channel instead of 3 channels).

2. Model Architecture 

The model used in this project was LeNet. It includes 5 layers. First three layers are convolutional layers. Each layer comprises of: (1) Setting random weights (tf.trunctated_normal) with mean of zero and variance of 0.1, (2)Setting biases to zero (tf.zeros), (3)Performing convolution with Valid padding (tf.nn.conv2d), (4) Performing Relu Activation (tf.nn.relu), (5)Max Pooling(tf.nn.max_pool).  Subsequently, the result was flattened using flatten function. Next layers are basic linear classifiers. Moreover, dropout was implemented in both layers with a probability of 0.67 in order to make the model more accurate. Factors from 0.6 to 0.7 have been tested and value around 0.65-0.67 proved to be the most effective. 

3. Model Training

The model was trained using Adam Optimizer from LeNet. The number of epochs used was 30 to allow the training accuracy to converge to the highest value. The learning rate was decreased to 0.001 which proved to give satisfactory results. The batch size was constrained by my memory (128). The probability used in the dropout was 0.67 but any value between 0.65-0.67 was giving similar results. 

4. Solution Approach

The objective of this project is to make a model as accurate as possible. The LeNet model and changes implemented, described in the previous section, enabled to reach the validation accuracy of 95%. The main objective for me was to reach high training accuracy which accounted for 0.997 and then stagnated. Subsequently, the model was tested using validation test. When the accuracy reached acceptable result (0.93< approx. 0.95) the model was tested with a testing set which accounted for 0.934. The model could be further improved by employing L2, expanding the training dataset and exploring more modern models. However, for a first big deep learning model, I believe that it is a satisfactory result. 

5. Final Results

Training Accuracy: 0.997  
Validation Accuracy: 0.951 
Test Accuracy: 0.93

6. Test a Model on New Images

The images chosen from Google Photos Collection are uploaded in the folder and represent turn right ahead (33), speed limit (70km/h) (4), speed limit (30km/h) (1), yield (13) and no entry (17), respectively. I believe the biggest challenge for the model, as it is proved below, is to tell the apart speed limits as they have similar characteristics where other signs have distinctive features.  
 
The accuracy accounted for 0.8 and the respective top 5 softmax probabilities and expected labels are in the IPython Notebook.

Even though the accuracy accounted for 80%, it should be highlighted that the model has problems identifying speed limits and one correct guess for speed limit 30 km/h (1) could easily be lucky (lower probability of 0.99). This is understandable as all signs have characteristic shapes and colours whereas each speed limit is similar and the only characteristic telling it apart is one number. That is why it should be considered more as 60%. Moreover, the training set could be enlarged with more speed limits photos as it looks like the biggest flaw of the model. Based on the probabilities, it can be deduced that model had no problems with other sings as it accounted for the probability of 100% in each case.  
 
 

