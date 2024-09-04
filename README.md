# Human-Activity-Recognition-in-Videos-using-CNN-LSTM-in-Sports-Domain

ABSTRACT
The act recognition (HAR) collects events to differentiate because of the sequence of annotations to recognise the actions of subjects to see the ecological situation. Humans have the flexibility to acknowledge a happening from one movement. It's the natural tendency of kinsmen to administer more attention to dynamic objects than to static objects. Human motion analysis is currently one in every of the foremost active research topics in machine learning. During this analysis the machine learning techniques for human action recognition provides detail issues, there has been an influx to the recent situation from the effective extraction and learning from live datasets as information. The methodology differs from traditional algorithms to this machine learning techniques uses hand-crafted heuristically derived features to the newly generated hierarchical based self-evolving features. Different types of quantitative and statistical tools are available for prediction and thus results are evaluated with various existing methods to induce better results of recognition. These techniques are classified into statistical forecasting models, shallow machine learning models, ensemble learning models, deep learning models and other learning models. From the literature review this work produces depth analysis results of deep learning models which are found to produce improved accuracy. Convolutional neural networks (CNNs) are extensively applied for image recognition problems giving state of-the-art results on recognition, detection, segmentation and retrieval. During this work we propose and evaluate several deep neural network architectures to mix image information across a video over a certain time period. We propose two methods capable of handling full length videos. The primary method explores various convolutional temporal feature pooling architectures, examining the assorted design choices which require to be made when adapting a CNN for this task. The second proposed method explicitly models the video as an ordered sequence of frames. For this purpose we employ a recurrent neural network that uses Long Short-Term Memory (LSTM) cells which are connected to the output of the underlying CNN. Recognizing human activities from video sequences or still images may be challenging task because of problems, background clutter, partial occlusion, Changes in scale, viewpoint lighting and appearance. Many applications including video surveillance systems, human-computer interaction, and robotics for human behavior characterization require a multiple activity recognition system.

INTRODUCTION

In recent years, human motion recognition has become a hot stock within the field of the applying system and academic research. As early as 1973, a psychologist named Johansson distributed the motion perception experiment of moving light spots, which is the first modern research on human motion recognition. Since then, until the 1990s, people began to pay more attention to the current field. So far, many researchers around the world have done plenty of research on human motion recognition technology. The standard research on human motion recognition may be divided into the subsequent parts, representation of motion information and recognition and classification of motion information.

To process the body action data, machine-learning algorithms are used to construct models and distinguish new data, like Support Vector Machines (SVMs), Hidden Markov Model (DBN-HMM), and deep learning (DL). As for application, visual communication plays a major role in people's communication. A far better understanding of body actions will increase communication efficiency. Human-computer interaction during this field refers to a machine's understanding of human behaviors through body actions. As an example, somatosensory game machines provide a much better experience for the players by capturing the players' actions in 3D space DL has been widely employed in image recognition, classification, evaluation, and predictive analysis in computer vision. It can directly extract information from the first data and form a big feature expression. First, the first data are pre-processed, and also the data features are extracted by hierarchical forward propagation and back propagation (BP). Each layer's expression is abstracted so the ultimate expression can better describe the computer file. A DL algorithm has advantages in action recognition and good modelling capabilities. It can process various input features and establish the connections between adjacent times to extract the action's context information without assuming the action features' distribution. Therefore, it will be applied in action recognition. Deep learning models offer lots of assurance for time arrangement determining, for instance the programmed learning of transient reliance and therefore the programmed treatment of worldly structures like patterns and regularity.

Video analysis provides more information to the popularity task by adding a temporal component through which motion and other information may be additionally used. At the identical time, the task is way more computationally demanding even for processing short video clips since each video might contain hundreds to thousands of frames, not all of which are useful. A naive approach would be to treat video frames as still images and apply CNNs to acknowledge each frame and average the predictions at the video level. However, since each individual video frame forms only a part of the video's story, such an approach would be using incomplete information and will therefore easily confuse classes especially if there are fine-grained distinctions or portions of the video irrelevant to the action of interest. Therefore, we hypothesize that learning a world description of the video's temporal evolution is very important for accurate video classification. This can be challenging from a modeling perspective as we've to model variable length videos with a hard and fast number of parameters. We evaluate two approaches capable of meeting this requirement: feature pooling and recurrent neural networks. The feature pooling networks independently process each frame employing a 1 CNN and then  combine frame-level information using various pooling layers. The recurrent neural architecture  we employ springs from Long Short Term Memory (LSTM) units, and uses memory cells to store, modify, and access internal state, allowing it to get long range temporal relationships. Like feature-pooling, LSTM networks care for frame-level CNN activations, and may learn the way to integrate information over time. By sharing parameters through time, both architectures are ready to maintain a continuing number of parameters while capturing a world description of the video's temporal evolution.

In order to find out a worldwide description of the video while maintaining a low computational footprint, we propose processing just one frame per second. At this frame rate, implicit motion information is lost. To compensate, following we incorporate explicit motion information within the style of optical flow images computed over adjacent frames. Thus optical flow allows us to retain the advantages of motion information (typically achieved through high-fps sampling) while still capturing global video information. Our contributions are often summarized as follows:
1. We propose CNN architectures for obtaining global video-level descriptors and demonstrate that using increasing numbers of frames significantly improves classification performance.
2. By sharing parameters through time, the amount of parameters remains constant as a function of video length in both the feature pooling and LSTM architectures.

PROPOSED METHOD

Fig. 1. The overall system architecture of the proposed human activity recognition.




3.1 PREPROCESSING
The input videos are first converted to a set of frames, each of which is represented by a matrix. Then a frame selection algorithm is applied. This can be done by fixed-step jumps to eliminate similar sequential frames. Based on our experiment, selecting one frame in every ‘n’ frames will not significantly reduce the quality of the system.’n’ is decided by dividing the total number of frames by frame sequence length. Here frame sequence length = 20. In this case, 320x240x3 images will be converted to a 64x64x3 output image. Thus, the shape of the input examples fed to the network, samples 6 signal components that are then arranged into a 64 × 64 single-channel, image-like matrix. Therefore, the input of the network has a shape 64×64×3, where 3 is the number of channels. The frame length specified is 20 frames per video that will be fed to the model as one sequence. One frame will be selected in every ‘n’ number of frames. Extraction occurs after resizing and normalizing the frames.

3.2 SPATIAL FEATURES EXTRACTION
Convolutional neural network (CNN) is a type of artificial neural network (ANN) based on convolution operation which is the core of extracting features. Convolution methods are of three types mainly, 1D convolution, 2D convolution and 3D convolution. Among them, 2D convolution and 3D convolution can often be used for feature extraction in action recognition. We will be using 2D convolution. Generally, a convolutional layer processes the input image and produces a batch of 2-dimensional feature maps containing spatial features, where the pooling layer is scaling down the extracted feature maps by applying down-sampling operations (i.e. max pooling, min pooling, or average pooling operations). The normalization layer is usually used before the activation function that normalizes the input values and gives more accurate activation.
After the input layer, three convolutional layers interleave with three max pooling layers. These are the basic 3 layers in ConvNet, i.e. the convolution layer and the max - pooling layers. The depthwise convolution operation generates multiple feature maps for every input channel, with kernels of size 3 × 3 for all the layers. The input of every convolutional layer is padded properly so that there is no loss of resolution from the convolution operation. The three max pooling layers use kernels of size 4 × 4. The ReLU function is used as an activation function within the whole network, while the loss is calculated with the cross entropy function. The Adam optimizer is used as a stochastic optimization method to optimize a categorical cross-entropy loss function. Additionally, to reduce possible overfitting, dropout layers are regularized during the training phase, with a 0.25 probability of keeping each neuron. The last stage of CNN has a flatten layer collapses the spatial dimensions of the input into the channel dimension so as to be passed to the dense layer.
We select a set of hyperparameters that are kept constant for all the activity groups and sensor configurations, based on literature best practices and empirical observations. We use a batch size of 4, as we find this value to speed up the learning process when compared with smaller sizes, without being computationally too complex to manage. The number of training epochs varies from 30 to up to 35, according to the behavior of individual configurations. The initial learning rate is fixed to  0.001. The network is implemented with the TensorFlow framework (Keras 2.7). After finishing the previous steps, we're supposed to have a pooled feature map by now. We flatten the pooled feature map into a column. The reason we do this is that we're going to need to insert this data into an artificial neural network (ANN) later on.

3.3 TEMPORAL FEATURES EXTRACTION
Despite the robustness and efficiency, CNN-based approaches can only be used for fixed and short sequence classification problems and aren't recommended to use for long and complicated statistical data problems. Mostly the problem is having sequential analysis over time. Generally, the RNN network analyzes the input hidden sequential pattern by concatenating the previous information with current information from both spatial and temporal dimensions and predicts the future sequence. We use LSTM for two main reasons: 
1. As each frame plays a very important role in a video, maintaining the important information of successive frames for a protracted time will make the system more efficient. The “LSTM” method is acceptable for this purpose. 
2. Artificial neural networks and LSTM have greatly gained success for the processing of sequential multimedia data and have obtained advanced results which end up in speech recognition, digital signal processing, image processing, and text data analysis.
In contrast to max-pooling, which produces representations which are order invariant, we propose using a recurrent neural network (RNN) to explicitly consider sequences of CNN activations. Since videos contain dynamic content, the variations between frames may encode additional information which might be useful in making more accurate predictions. We capture dependencies among time frames using an RNN encoder architecture that learns temporal information from the input time-series data. 
The output data from the flatten layer passes through a unidirectional LSTM layer to better extract the temporal features in the sequence data. LSTM has 32 memory cells(32 neurons). The softmax function will return the foremost likely class of the input windows in the multi-class classification task. The Softmax classifier converts the output of the upper layer into a probability vector whose value represents the probability of classes to which this sample belongs. 


3.4 Proposed CNN-LSTM Model





Fig. 2. An illustration of the proposed hybrid network for sports activity detection.






Fig.3. An illustration of parameters of the model layers
In this study, a combined method was developed to automatically detect the sports cases using different types of videos. The structure of this architecture was designed by combining CNN and LSTM networks, where the CNN is used to extract complex features from images extracted from the video datasets and LSTM is used as a classifier.Above figure illustrates the proposed hybrid network for human activity detection.The network has 11 layers: three convolutional layers, three pooling layers, three dropout layers, one LSTM layer, and one output layer with the softmax function. Each convolution block is combined with 2D CNNs and one pooling layer, followed by a dropout layer characterized by a 25% dropout rate. The convolutional layer with a size of 3 × 3 kernels is used for feature extraction that is activated by the ReLU function. The max-pooling layer with a size of 4 × 4 kernels is used to reduce the dimensions of an input image. In the last part of the architecture, a function map is transferred to the LSTM layer to extract time information. After the convolutional block, the output shape is found to be (none, 2, 2, 64). The input size of the LSTM layer has become (256). After analyzing the time characteristics, the architecture sorts the images through a dense layer to predict whether they belong under any of the three categories (baseball, basketball, and volleyball).

DATASETS USED

4.1 Training Dataset:
UCF-50 Dataset
The UCF-50 contains unconstrained web videos of 50 action classes with more than 100 videos for each class.The UCF-50 dataset has over 6675 videos in the sports category. The videos are captured in different lighting conditions, gestures, and viewpoints. One of the major challenges in this dataset is the mixture of natural realistic actions and the actions played by various individuals and actors, while in other datasets, the activities and actions are usually performed by one actor only. Therefore, UCF50 is considered as a very comprehensive dataset. The UCF Sports action has been used for benchmarking purposes based on temporal template matching.
UCF50 action recognition dataset contains:
50 action categories 
133 average videos per action category 
199 average number of frames per video 
320 average frames with per video 
240 average frames height per video 
26 average frames per second per video
From these 50 categories we have considered 3 categories for our project. Those are:-
Baseball
Basketball
Volleyball
Figure    shows a sample frame of three different video clips and actions of the classes chosen from the UCF50 dataset.


4.2 Testing Dataset:

These videos are taken from YouTube which are totally unseen and are untrained on the model before.
With the help of these videos we can measure the accuracy of the model on unknown videos
These videos are:
8 - 15 seconds long
24 total videos combining all three categories
320 average frames width per video
240 average frames height per video
25 average frames per second per video

5. EXPERIMENTAL RESULTS

All the experiments in this paper were implemented in Python using Keras with TensorFlow backend on NVIDIA P100 GPUs

Experimental results analysis 

5.1 Experimental setup 
In the experiment, the dataset was split into 75% and 25% for training and testing, respectively. The results were obtained using a cross-validation technique. The proposed network consists of 3 convolutional layers, the learning rate is 0.001, and the maximum epoch number is 70, as determined experimentally. The CNN-LSTM network was implemented using Python and the Keras package with TensorFlow2 on an Intel(R) Core(TM) i3-2.2 GHz processor. In addition, the experiments were executed using the graphical processing unit (GPU) with 12 GB(4 GB internal).

5.2 Results and its analysis
Hyperparameter tuning
In machine learning, hyperparameter optimization or tuning is the process of choosing a set of optimal hyperparameters for a learning architecture. A hyperparameter is a parameter whose value is identified to control the learning process. By contrast, the values of other parameters are learned for an optimized network. The following tables give an overview of the hyperparameters.

Table 1. Hyperparameter tuning for patience
Patience
Training time (ms)
Validation loss
Accuracy
10
281
0.3951
0.9010
15
304
0.3220
0.9505
20
284
0.3891
0.9406


Table 2. Hyperparameter tuning for epoch
Epoch
Training time (ms)
Validation loss
Accuracy
20
271
0.4463
0.9208
30
301
0.4391
0.9505
40
291
0.2971
0.9208



Table 3. Hyperparameter tuning for batch-size
Batch-size
Training time (ms)
Validation loss
Accuracy
3
262
0.3425
0.9505
4
269
0.3314
0.9505
5
345
0.3904
0.9307
















  (a)                                                                               (b)
Fig. 4. Evaluation metrics of sports activity detection system based on CNN-LSTM
Architecture (a) Accuracy (b) Loss


Before explaining our hyper parameter tuning approach, it's important to elucidate a process called "cross-validation" because it is taken into account as a crucial step within the hyper parameter tuning process. Cross-validation (CV) could be a method to estimate the accuracy of machine learning models. Once the model is trained, we can't be certain of how well it'll work on data that haven't been encountered before. Assurance is required regarding the accuracy of the prediction performance of the model. To gauge  (a)the performance of a machine learning model, some unseen data are needed for the test supported the model's performance on unseen data, we are able to determine whether the model is under fitting, over fitting, or well-generalized. Cross-validation is taken into account a really helpful technique to check how effective a machine learning model is when the info in hand are limited To perform cross-validation, a subset of the info should be put aside for testing and validating this subset won't be accustomed train the model, but rather saved for later use K-Fold is one among the foremost common techniques of cross-validation, and it's also the cross-validation                                           (b) technique. In K-Fold cross-validation, the parameter K indicates the quantity of folds or                                      
Sections that a given dataset is split into.The importance of hyper parameters lies in their ability 
to directly control the behaviour of the training algorithm. Choosing appropriate hyper parameters plays a really important role within the performance of the model being trained. It's important to possess three sets into which the information are divided, i.e., a training, testing, and validation set, whenever the default parameter is altered so as to get the required accuracy, so on to prevent data leaks. Thus, hyper parameter tuning will be simply defined because of the process of finding the simplest hyper parameter values of a learning algorithm that produces the most effective model. The optimum values of the hyper parameters are given within the above tables in bold.

Table 4. CNN-LSTM performance for different frame sequences
Frame sequence
Accuracy (%)
Precision (%)
F1-score (%)
Recall (%)
20
93
93
93
93
30
92.08
91.9
92
92
40
96.04
96.14
96.03
96.04
50
95.28
95.16
95.04
95.02
60
94
94.02
94
94.01



Furthermore, Table4 shows the accuracy of each class of the developed CNN-LSTM network for the training and testing dataset.The testing data was taken from YouTube videos which are completely unseen and untrained. 

Table 5. Summary of training and testing datasets category-wise


Baseball
Basketball
Volleyball
Training dataset
97.4
93.5
96.7
Testing dataset
85.7
57.1
57.1




Confusion Matrix
Fig. 5 depicts the confusion matrix of the test phase of the proposed CNN-LSTM architecture for human activity classification. Among the 400 videos, around 20 were misclassified by the CNN-LSTM architecture. It was found that the proposed CNN-LSTM network has better and consistent true positive and true negative values and lesser false negative and false positive values. Therefore, the proposed system can efficiently classify between the three classes. Further, Fig. 4 depicts the performance evaluation of the CNN-LSTM classifier graphically with accuracy and cross-entropy, i.e loss in the training and validation phase.The obtained training and validation accuracy is 96.04% and 95.08%, respectively, at epoch 30. Similarly, the training and validation loss is 0.05 and 0.07, respectively, for the CNN-LSTM architecture. Better scores of training and validation accuracy were achieved using the CNN-LSTM architecture.





                (a)                                                                          (b)
Fig. 5. Confusion matrix of the proposed sports activity detection system CNN-LSTM


Performance evaluation metrics 
The following metrics are used to measure the performance of the proposed system: TP denotes the correctly predicted Baseball videos, FP denotes the Basketball videos or volleyball videos that are misclassified as
Baseball by the proposed system, TN denotes the basketball or volleyball videos that are correctly classified, and FN denotes the baseball videos that are misclassified as basketball or volleyball cases. 
Accuracy = (TP + TN)(TN + FP + TP + FN) 

Sensitivity = TP(TP + FN) 

Specificity = TN(TN + FP) 

F1 − score = (2*TP)(2*TP + FP + FN)

Recall = TPTP + FN

Precision = TPTP + FP

F-1 score is the weighted average of both precision and recall and it is used in case the data is imbalanced. On the other hand, the indices 'accuracy estimates the performance of the classifier in a better manner if there is a similar cost for FP and FN.

For instance, the average accuracy of CNN for all types of sequences is 97, LSTM is 78.57 and proposed CNN-LSTM achieved 96 average accuracy.
Below table shows the accuracy of our hybrid approach as compared to other deep learning models. 


Table 6. Performance of different models with the proposed CNN-LSTM model
Model 
Accuracy(%)
Precision(%)
Recall(%)


F1 Score(%)


Training time
(min)
CNN
97
97
97
97
180
LSTM
78.57
78.06
78.57
78.29
150
LRCN
96.04
96.14
96
96
20



The proposed model achieved the very best accuracy as compared to solo deep learning-based models and traditional machine learning models. The main reason behind the very best performance of the proposed model is learning spatial and temporal information from the computer file while other models only extract one kind of feature at a time. The frames sequence also depends on the accuracy of the model, if we select a awfully large frames sequence then it can decrease the model accuracy and performances. We used different optimizers and after investigating all optimizers we selected the "Adam" Optimizer for our experiments. All the experiments are performed using the identical hyperparameters like batch size = 4, learning rate = 0.001 and epoch = 30. These optimal parameters are selected after performing an oversized number of experiments on different parameters. Our model gave a wonderful performance on these parameters, so we chose these parameters the very best
accuracy of 96.04% is achieved by the CNN-LSTM hybrid model on a 40 frames sequence. The second highest accuracy is achieved on the 50 frames sequence. Table 7 and 8show the performance evaluation metrics on training and testing datasets category-wise. 


Table 7. Performance of the proposed model on UCF-50 dataset


Accuracy 
Precision 
Recall 
F1 score
Baseball
97.43
95
97.44
96.20
Basketball
93.54
96.67
93.55
95.08
Volleyball
96.77
96.77
96.77
96.77



Table 7. Performance of the proposed model on YouTube dataset


Accuracy 
Precision 
Recall 
F1 score
Baseball
87.5
70
87.50
77.78
Basketball
50
80
50
61.54
Volleyball
62.5
55.56
62.50
58.82


6. CONCLUSION

The key to the success of human motion recognition is to capture the spatiotemporal motion patterns of all parts of the organic structure at the identical time. During this presentation, a deep learning method supporting the motion characteristics of local joints is proposed to acknowledge the motion samples. The effectiveness of the tactic is evaluated on two datasets. Aiming at the shortcomings of the model. The common error of the model is reduced and also the spatial configuration information is introduced to boost the popularity accuracy of the model. It has been observed that CNN with LSTM models can accurately identify the activity present as compared to RNNs, which lags the convolution method of the video data. The quantity of activities in each video and therefore the sub-actions is additionally immediately discovered. The state-of-the-art localization efficiency has been demonstrated on standard action datasets including temporal annotations for the action segment represented within the pipeline, start time and end time are highlighted for the identical activity of a definite person or different activity of assorted persons present within the video. The proposed approach has the potential to locate the precise temporal boundary of the instance of operation and takes into consideration the interdependency of the segments of the action instance. As discussed the proposed solution outperforms the efficiency of the state-of-the-art recognition methods for the sports domain dataset that has broadly three activities namely baseball, basketball and volleyball We presented two- video-classification methods capable of aggregating frame-level CNN outputs into video-level predictions. Feature Pooling methods which max-pool local information through
Time and LSTM whose hidden state evolves with each subsequent frame. Both methods are motivated by the thought that incorporating information across longer video sequences will enable better video classification. If speed is of concern, our methods can process a whole video in round. Training is feasible by expanding smaller networks into progressively larger ones and fine-tuning. The resulting networks achieve state-of-the-art performance on the UCF-50 benchmarks, supporting the thought that learning should occur over the complete video instead of short clips. Additionally we explore the need of motion information, and make sure that for the UCF-50 benchmark, so as to get state-of-the-art results, it's necessary to use optical flow.


7. REFERENCES
1. "Long-term Recurrent Convolutional Networks for Visual Recognition and Description" by Jeff Donahue, Lisa Anne Hendricks 
2. "Vision-based human activity recognition: a survey" by Djamila Romaissa Beddiar in Multimedia Tools and Applications 
3. "Human Activity Recognition: A Survey" by Charmi Jobanputra in Procedia Computer Science 
4. "Human activity recognition with smartphone sensors using deep learning neural networks" by Charissa Ann Ronao in Expert Systems with Applications

