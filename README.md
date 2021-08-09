# ANN_Practice with Tensorflow
### 인공신경망과 딥러닝 실습 과제 모음   

## 1. Tensorflow 에 관하여
### 1.1 Tensorflow 란?
텐서플로(TensorFlow)는 다양한 작업에대해 데이터 흐름 프로그래밍을 위한 오픈소스 소프트웨어 라이브러리이다. 심볼릭 수학 라이브러리이자, 인공 신경망같은 기계 학습 응용프로그램에도 사용된다. 이것은 구글내 연구와 제품개발을 위한 목적으로 구글 브레인팀이 만들었고 2015년 11월 9일 아파치 2.0 오픈 소스 라이선스로 공개되었다.

### 1.2 Tensorflow의 특징
```c
데이터 플로우 그래프를 통한 풍부한 표현력
아이디어 테스트에서 서비스 단계까지 이용 가능
계산 구조와 목표 함수만 정의하면 자동으로 미분 계산을 처리
Python, C++, Go, Java, R[2]을 지원하며, SWIG를 통해 다양한 언어 지원 가능  
```

### 2. 과제 내용

#### 2.1 HW1 : Python Practice
+ Problem 1: print out all prime numbers between 2 and 100. 
  - 2부터 100까지에서 소수(prime number)만 출력하기
  - Use for and if statement. If you’re familiar to python, you can use different routine.

+ Problem 2: load the given csv file into a numpy variable, and calculate the sum of each column and each row. 
  - 주어진 csv화일을 읽어서 numpy 배열에 넣고, 각 행의 합과각 열의 합을 구하라.
  - Use numpy.loadtxt

#### 2.2 HW2 : Breast Cancer Wisconsin Dataset 
+ About Dataset
  - **Classification problem**
    - 10 input variables
    - 1 binary output variable (benign or malignant)
  - Originally hosted by UCI
  - 569 data samples
    - Use the first 100 samples as test set
    - Use the next 100 samples as validation set
    - Use the others as training set
    
+ About Data Preparation
  - Download breast-cancer-wisconsin.data
  - Remove the rows with missing values “?”
    - With any text editor
  - Load it in the python
  - Drop the first column: 
    - The first column is ID, which does not carry any information about the tissue. 
  - **Normalize the input variables.**
  - Set the output variable
    - Set Malignant: 1, benign: 0
  - Data split: train, test, & validation set
 
+ Basic Model
  - Model Structure
    - 9 inputs
    - 10 hidden neurons with ReLu activation functions
    - 1 output neuron with sigmoid activation function.
  - Compile and learning condition
    - Optimizer=rmsprop, 
    - Loss function=binary crossentropy
    - Epochs=200
    - Batch_size=10
    - EarlyStopping with patience=2

+ Problem1 : Show your code.
  - 0.5 Points for data preparations
    - 1. data load
    - 2. data normalization
    - 3. output coding
    - 4. data split
  - 0.5 Points for the model definition and learning
    - 1. model definiton
    - 2. model setup (loss, optimizer)
    - 3. correct fitting procedure
    - 4. correct evaluation procedure
    
+ Problem 2 : Repeat training of the model 5 times, and collect their losses and accuracies using the table
  - Are they all consistent ober trials? It not, why?

+ Problem 3 : Investigate whether the activation function of the hudden layer affects the accuracy
  - Still the same model (the # of hidden neurons : 10)
  - 4 different cases : None, Relu, sigmoid, tanh
  - For each case, repeat training 10 times and report the mean and standard deviation of loss and accuracy in the training and test data set.
  - Use the similar table in the problem #2.
  - Which one is the best? Why? 
  
+ Problem 4 : 
  - Let’s investigate how the number of hidden neurons affects the performance.
    - Set the activation function of the hidden layer to Relu.
  - Change # of hidden neurons systematically, and then re-training the model.
    - Collect the data and construct the table for the following # of hidden neurons: 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000.
    - For each case, repeat training 5 times and report the mean and standard deviation of loss and accuracy in the training and test data set. 
  - What is the best case? Why did you select it? (i.e. which one did you use among 4 metrics you collected?)
  
+ Extra Question 1 : 
  - Generally, after performing the rough search we did in the Q4, we performed the more fine-tuned search for the optimal # of hidden neurons.
    - As an example, if we found that the best performance was achieved near 20~50, we performed another experiment varying # of hidden neurons: 25, 30, 35, 40, 45, and select the case with the best performance.
  - The question is “why didn’t we try all cases at once?”
    - As an example, we can try for all cases: 2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60 ….
    - But we don’t. Why?
    
+ Extra Question 2 : 
  - After learning, we can analyze the learned weights
  - Construct a model without a hidden layer; all input units are directly corrected to the output.
  - After learning, using the following commands, you can get the weights and bias.
    ```c
    - w=model.get_weights()[0]
    - b=model.get_weights()[1]
    ```
  - Please analyze the model based on the learned weights. What does the large weight mean? What does the weight near zero mean? What does the negative value mean?
    - Check breast-cancer-wisconsin.names.
    

#### 2.3 HW3 : Chest X-ray (Pneumonia - 폐렴)

+ About Project
  - ***Classification problem***
    - input variable: images! 
    - 1 binary output variable (pneumonia or normal)
  - 5863 x-ray images
    - Already split into train, validation and test.
    - https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
  
+ Global Average Pooling 2D (GAP) 
Each response map is averaged into single neuron. Thus, the GAP output of n feature maps is n neurons regardless the width and heights of feature maps

+ Problem 1 : Show your codes, accuracy, and loss in the training and test set, also show the accuracy graph and loss graph in the training and validation set.
  - Data preprocessing: The image sizes all vary. Thus, resizing is essential.
    - When loading images, resize the image into [128, 128]
    - flow_from_directory(train_dir, target_size=(128,128), batch_size=20,class_mode='binary’)
  - Base model: you will use a pre-trained model, VGG16 (weights=‘imagenet’).
  - Classifier: the top MLP structure should be:
    - GlobalAveragePooling2D -> Dense(512) -> BatchNormalization -> Activation(Relu) -> Dense(128) -> Dense(1) 
  - You should do 2-step fine-tuning
    - 100 epochs for the frozen base + 50 fine-tuning epochs (only tune 5-blocks)
    - Learning parameters: RMSprop with learning rate of 1e-5 
    - When you load a model, you should set optimizer again.
  - You can run multiple times and average the results. However, I do not recommend, since it will take quite long time to learn (more than 2 hours using a GPU)
  
  ***Note. Do not forget saving the learned model before and after fine-tuning. You will use the saved model in QE2***
  
+ Problem 2 : Show your codes, accuracy, and loss in the training and test set, also show the accuracy graph and loss graph in the training and validation set.
  - The previous model showed serious overfitting. Thus, let’s add dropout.
  - The modified classifier: the top MLP structure should be:
    - GlobalAveragePooling2D -> Dropout(0.25) ->Dense(512) -> BatchNormalization -> Activation(Relu) -> Dropout(0.25) -> Dense(128) -> Dropout(0.25) -> Dense(1)
  - You should do 2-step fine-tuning
    - 100 epochs for the frozen base + 100 fine-tuning epochs
    - All other parameters should be same with the problem 2’s
    ***Note. Do not forget saving the learned model before and after fine-tuning. You will use the saved model in QE2***
    
+ Problem 3 :
  - Repeat Q2 with image resizing into [256, 256] and [512, 512].
    - For [512, 512], due to memory limitation, you should change the batch size into 10. 
    - For [256, 256], the batch size of 20 is okay. (No change is required)
  - You should do 2-step fine-tuning

+ Problem 4 : 
  - Using the Chapter 5.3 of the textbook, draw the area that was important for classification.
  - You can use matplotlib’s pyplot.imshow.

+ Extra Question 1 : 
  - We will try CNN for varying image sizes. 
  - For data_flow_directory, do not specify resize. In other words, simply
  ```c
  flow_from_directory(train_dir, batch_size=20,class_mode='binary’)
  ```
  - Instead, the input_shape of CNN should be specified as [None, None, 3]
  ```c
  input_shape = [None, None, 3]
  ```
  - You should do 2-step fine-tuning
  - *Run the code. Does it work?*
  - *Replace GlobalAveragePooling2D with Flatten. Run the code, does it work?*
  - *Does it work better than the best model of Q3? If so, why? If not, why?*

+ Extra Question 2 : 
  - There are other methods to evaluate the model.
  - Compute the following scores in Q1~Q3 (and QE1).
    - Precision, Recall (sensitivity), Specificity, F1 score, AUC
    - These scores should be computed in the test data set only.
  - You need to use sklearn (adapt it into your code!)
  ```c
  y_pred=model.predict_generator(test_generator)
  matrix = sklearn.metrics.confusion_matrix(y_test, y_pred>0.5)
  auc=sklearn.metrics.roc_auc_score(y_test, y_pred)
  ```
  - Which model was the best considering all of the computed scores? 
  
+ Extra Question 3 : 
  - Let’s use different CNN base model, inceptionV3(weights=‘imagenet’).
    - Use the same decision maker part with the models in the main questions
  - Following Q1-Q3, and QE1, QE2, find the best model. The model should be tested through
    - 2-step fine-tuning (Q1)
    - Avoiding overfitting (Q2)
    - Investigating whether image resizing affects the performance (Q3 and QE1)
    - Various evaluation methods (QE2)

#### 2.4 HW4 : Colorization

+ Problem 1 : Colorization with U-net
  - You’re going to do colorization example over CIFAR10 dataset in the U-net slides.
  - Use the data class code (U-net(2).pdf, slide 10).
  - Use the code in the U-net class code (U-net(1).pdf).
  - You don’t have to change any parameter of the example code.

+ Problem 2 : Colorization with a deeper U-net
  - Can you add one more downsampling encoding block (i.e. One more encoding with the conv function)
    - The number of filters will be doubled for every downsampling.

+ Problem 3 : Colorization with SCAE
  - Let’s try a different structure, SCAE. 
  - Change the U-net class. Just cut-off skip connections, and related concatenation.
    - Hint. You should modify deconv_unet function.

+ Extra Question 1 : Colorization with deeper U-net
  - Extending Q2, can you add more downsampling encoding blocks infinitely?
  - If not, why?
  
#### 2.5 Practice 6 : IMDB Text Data
+ Problem 1 : Please copy and paste the decoded review of the Xth sample to the blank below.
+ Problem 2 : Capture the screenshot of the following codes, and submit it.
```c
word_index_of_X=train_data[X][0:5]
print(word_index_of_X)
print( [ reverse_word_index.get(i-3,'?') for i in word_index_of_X ] )
print( x_train[X][word_index_of_X[1]] )
print( x_train[X][word_index_of_X[3]] )
```
+ Problem 3 : Please explain the text data processing in the IMDB database.Based on the previous practice, what does the vectorizedinput mean? 
+ Extra Question 1 : 
  - Read the decoded review of the Xth sample. What do you think? Is this positive? Or negative?
  - Compare it with your NN model’s results.
    - What is the output of the NN model? Provide the screenshot how you check the results of NN model.
    - Is it consistent with your thought?
    
#### 2.6 Practice 7 : Red wine Quality
+ Problem 1 : Show your codes
  - Model : 
    - hidden layer 3개를 가지는 네트워크를 구성, 각 히든 레이어의 히든뉴런 개수는 512개로 할것
    - Optimizer는 adam, loss는 MSE로 설정
  - Data preparation : 
    -  train set은 data의 첫 1000개를 사용하도록 하고
    - test set은 data의 1000개 이후의 데이터를 사용하도록 한다
    - min-max Normalization을 시행. data를 0과 1사이로 바꾸도록 함 (hint. preprocessing.MinMaxScaler)
  - Learning : 
    - 학습과정에서 vadliation은 20% hold-out CV를 이용하도록 함
    - epoch은 500으로 설정. batch size는 default
  - Evaulation :
    - MAE를 기준으로 결과를 제시함 (trainig set, validation set, test set에서의 MAE값 제시)
    - MAE의 learning curve (training set and validation set)를 제시함
    - Test set의 첫 10개의 prediction결과와 실제결과를 확인.

+ Problem 2 : Show your results (learning curve, evaluation in the training set, validation set & test set, prediction in the test set)
+ Problem 3 : weight regularization
  - 모든  Layer에 L1 regularization 0.001, L2 regularization 0.001을 적용하고, 다시 코드를 실행해보자.
+ Problem 4 : Explaining
  - 위의 두 결과를 비교하고, 이유에 대해 설명해보아라. 특히 training set에서의 MAE이 어느 쪽이 더 우수하며, test set에서의 MAE는 어느쪽이 더 우수한지 설명해야하며,  왜 이러한 결과가 나타났는지 설명하여야 한다

#### 2.7 Practice 8 : CIFAR10
+ Notice
  - You will tune hyperparameters of a CNN model for CIFAR10.
  - Use the codes in the lecture slide, “CNN(4)”.
  - I provided “chapter5_1_cifar10.py”.
  - Follow the instruction in this file, “step-by-step” and collect the results.

+ Problem 1 : Show the modified “build_model” function.
  - We will add “Dropout” layers after the max-pooling layer and the hidden layer of MLP. 
  - Add the following lines to the right locations.
  ```c
  model.add(layers.Dropout(0.25)) # after max-pooling
  model.add(layers.Dropout(0.5)) # after the hidden layer of MLP
  ```
+ Problem 2 : 
  - When we use the dropout, we need more time to learn the model. So, please increase the number of maximum epochs to 50.

+ Problem 3 : 
  - The above learning curve is oscillating.
  - Maybe it’s better to decrease the learning rate. Change thelearning rate of RMSprop to 0.0001.
  - Also, the smaller size of minibatch also helps to stabilize the learning procedure. Set minibatch to 32. (it will increase the learning time about three times)

+ Problem 4 : 
  - As I told before, the other approach of machine learning is “highcapacity model with strong regularization”.
  - Indeed, it is not recommended to add dropout layers to the small-sized NN model.
  - Do you think our model is large enough? Is there any chance that our model is underfitting currently?
  - To test it, let’s increase model capacity. (See the structure in the nextpage)
  - Since the model becomes more complex, we need more time to learn. Set the number of maximum epochs to 100.
  
+ Extra Question 1 : 
  - There is another possible way to have “high-capacity model”.
  - Let’s try to increase the number of hidden neurons in MLP four times (512) without adding convolution layers. 
  
#### 2.8 Practice 9 : Data augmentation + Transfer learning
+ Problem 1 : 
  - Using the code in the slide 19 of CNN(5), add “data augmentation” to the train_datagen.
  - Increase the number of epochs to “60”.

+ Problem 2 : 
  - Let’s add dropout layers to the model.
  - Let’s add dropout layers after every maxpool layers with 0.25 dropoutprobability. (Thus, we have 4 maxpool layers, you will have 4 dropout layers.)
  - Set the number of epochs to “100”.

+ Problem 3 : 
  - Chapter5_3.py uses “VGG16” as a pretrained model.
  - Let’s try to use “InceptionV3” instead of “VGG16”

+ Problem 4 : 
  - Run the code. Attach its loss graph. 
  - What is the test accuracy? Is it better than the results of Q2?

+ Extra Question 1 : 
  - Let’s do the 2-step fine-tuning.
  - Copy Chatper5_3.py into a new file.
  - Let’s load the saved weights of Q4’s learning, instead of build_model()
    - The code contains the statement for model saving.
    - model.save('cats_and_dogs_small_pretrained.h5’)
  - We will fine-tune only top 2 inception blocks. (we will freeze the first 249 layers, and unfreeze the rest.)
    - We note that “conv_base=model.layers[0]”
    - You may check “https://keras.io/applications/#inceptionv3” for this modification.

+ Extra Question 2 : 
  - Set the number of epochs to 50.

#### 2.8 Practice 10 : CNN-XAI
+ Problem 1 : CNN(7)강의의 17번 슬라이드의 코드를 이용해서 conv2d_3의 X번째 response map을 그리세요. (X는 여러분 학생증번호의 마지막 한자리)
```c
layer_outputs = [layer.output for layer in model.layer[:8]]
activation_model = models.Model(model.input, layer_outputs)
activations = activation_model.predict(img_tensor)
print(len(activations))
print(activations[0].shape)
plt.matshow(activations[0][0,;,;,19]
```

+ problem 2 : CNN(8)강의의 9번 슬라이드의 코드를 이용해서 blcok5_conv1의 filter weights를 그리세요.
```c
def deprocess_image(X):
...

def generate_patterns(layer_name, filter_index, size=150):
  layer_output = model.get_layer(layer_name).output
  loss = K.mean(layer_output[:,:,:,filter_index])
  grads=K.gradients(loss, model.input)[0]
  grads/=(K.sqrt(K.mean(K.square(grads)))+1e-5)
  iterate = K.function([model.input], [loss, grads])
  input_img_data = np.random.random((1,size,size,3)*20+128
  step = 1
  for i in range(40):
    loss_value, grads_value = iterate([input_img_data])
    input_img_data += grads_value * step
  img = input_img_data[0]
  return deprocess_image(img)
```
+ Problem 3 : 앞의 문제와 유사하게, filter weight를 그리게 될것입니다. 그런데 이번에는 VGG16의 convolution layer들을 그리는 것이 아니라 chapter 5.2 코드에서 사용한 모델을 이용할 것입니다. 이 문제를 포함하는 문제들에서 각각 conv2d_1, conv2d_2 그리고 conv2d_3에 해당되는 weight를 그려주세요.

+ Problem 4 : 위의 결과가 CNN(7)의 슬라이드 20-23과 다른가요? 다르다면 어떻게 다른지 설명해주세요.

+ Problem 5 : 다음의 구조를 models.Model() 을 이용하여 다시 작성하세요. 
```c
def build_model():
    model=models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                            input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=optimizers.RMSprop(lr=1e-4),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model
    ```

#### 2.9 Practice 11 : Data Collection

+ Problem 1 : In the following web-page, collect all abbreviated names of the “level B” conferences. https://jackietseng.github.io/conference_call_for_paper/conferences-with-ccf.html Show your code using BeautifulSoup.

+ Problem 2 : Show the parsing results.

+ Problem 3 : 	
  - There are only three species of elephants. African bush elephant, forest elephant (mostly in the sub-Saharan Africa), and Asian elephant. We generally called African elephants for the former two species. Also, the Asian elephants are mostly lived in India, and thus, they are also called as Indian elephants.

  - We are going to make a model to distinguish the Indian elephants from the African elephants. Using the google_image_download package introduced in the class, we try to construct the dataset. The number of images to collect is shown below.

    - Train set: 500 images for each group
    - Validation set: 200 images for each group
    - Test set: 200 images for each group
  - Modify the given code, accepting the number of images to download. And show your code. 
  
+ Problem 4 : Show a screen capture of the resultant directories in the linux File Explorer.

+ Problem 5 : 	
  - In the crawled results, are there any overlapped samples between the train, validation, and test set? 
  - If so, it is a serious problem: though the test set should not be seen during the training phase, since a part of the test set is a sub set of the training set, it is not possible.
  - If your method has this issue, please suggeste how you can remove the problem keeping the size of each set. 
  - If your method does not have this issue, why is it? please explain.

+ Problem 6 : 
  - Check through the dataset "visually". Can you see any weird image? Please attach the example.

#### 2.10 Practice 13 : Auto-Encoder

+ Problem 1 : 
  - We are going to apply AE to cifar10 dataset. 
    - Change the function data_load
    - in the function “main”: x_nodes=32*32*3
    - in the function __init__ of class AE: Loss function: set as “mse”
    - in the function “show_ae”: Reshape into (32,32,3) instead of (28,28)

+ Problem 2 : Attach the results of show_ae with 360 hidden neurons
+ Problem 3 : Attach the results of show_ae with 1080 hidden neurons
+ Problem 4 : 
  - Let’s work on SAE.
    -  Change the function data_load
    - in the function “main”: x_nodes=32*32*3
    - in the function __init__ of class AE: Loss function: set as “mse”
    - in the function “show_ae”: Reshape into (32,32,3) instead of (28,28)
+ Problem 5 : Attach the results of show_ae, where the model was initiated with z_dim=[320, 290]
+ Problem 6 : Compare two models and the model in Q1B in terms of reconstruction quality and validation loss, considering the number of tunable parameters

#### 2.11 Practice 14 : U-net

+ Problem 1 : 
  - We are going to apply SCAE to cifar10 dataset. 
    -  Change the function data_load (Hint. you don’t need to reshape the dataset.)
    - in the function “main”  (Hint. you need to update “input_shape”)
    - in the function __init__ of class AE (Hint. you also need to update the size of output layer besides the loss function)
    - in the function “show_ae” (Hint. you also need to update the size of codes.)

+ Problem 2 : Attach the results of show_ae
+ Problem 3 : Change the number of filters as follows, and attach the results of show_ae.(124, 64, 1, 16, 64, 124)
  - The results of the previous SCAE model (Q1B) were not good enough. 
  - There are two different ways of improvement. 
    - A. Increased the number of filters.
    - B. increased the size of codes. 
+ Problem 4 : Change the number of filters as follows, and attach the results of show_ae. (82, 64, 16, 16, 64, 82)
+ Problem 5 : Compare two models (Q2A and Q2B) in terms of reconstruction quality and validation loss, considering the number of tunable parameters
+ Problem 6 : 
  - Implement the code of U-net for simple reconstruction of CIFAR10 on the slides 7-16.
  - Set the following learning parameters: Batch_size=128, Epochs=200
  - Change the number of filters in order:  56, 32, 16, 32, 56

## 정리







