import numpy
import os
import cv2
from tqdm import tqdm
from random import shuffle

trainDir = r'C:\Users\Dev Parikh\Documents\PythonFiles\TensorFlow\data\Training'
testDir = r'C:\Users\Dev Parikh\Documents\PythonFiles\TensorFlow\data\Testing'
imageSize = 100
LearningRate = 0.001

Model_Name = 'tumorvsnormal-{}-{}.model'.format(LearningRate, '2-conv-basic-video')
#[1,0] - Cow
#[0,1] - Alligator
def LabelImage(img)  :
    labelWord = img.split()[-3]
    if labelWord == 'Tumor':
        #Alligator
        return [0,1]
    elif labelWord == 'Normal':
        #Cow
        return [1, 0]
    else: print("fail")

def createTrainData():
    trainingData = []
    for img in tqdm(os.listdir(trainDir)):
        label = LabelImage(img)
        path = os.path.join(trainDir,img)
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (imageSize, imageSize))
        trainingData.append([numpy.array(img),numpy.array(label)])
        shuffle(trainingData)
    numpy.save('trainingData.npy', trainingData)
    return trainingData

def processTestData():
    testingData = []
    for image in tqdm(os.listdir(testDir)):
        path = os.path.join(testDir,image)
        image_num = image.split()[0]
        image = cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(imageSize,imageSize))
        testingData.append([numpy.array(image),image_num])
    numpy.save('testing_data.npy',testingData)
    return testingData


trainingData = createTrainData()

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

NeuralNet = input_data(shape=[None, imageSize, imageSize, 1], name='input')
NeuralNet = conv_2d(NeuralNet, 32, 2, activation='relu')
NeuralNet = max_pool_2d(NeuralNet, 2)
NeuralNet = conv_2d(NeuralNet, 64, 2, activation='relu')
NeuralNet = max_pool_2d(NeuralNet, 2)
NeuralNet = fully_connected(NeuralNet, 1024, activation='relu')
NeuralNet = dropout(NeuralNet, 0.8)
NeuralNet = fully_connected(NeuralNet, 2, activation='softmax')
NeuralNet = regression(NeuralNet, optimizer='adam', learning_rate=LearningRate, loss='categorical_crossentropy', name='targets')
model = tflearn.DNN(NeuralNet, tensorboard_dir='log')


shuffle(trainingData)
train = trainingData[:-100]
test = trainingData[-100:]

X = numpy.array([i[0] for i in train]).reshape(-1,imageSize, imageSize, 1)
Y = [i[1] for i in train]

test_x = numpy.array([i[0] for i in test]).reshape(-1,imageSize, imageSize, 1)
test_y = [i[1] for i in test]

model.fit({'input': X}, {'targets': Y}, n_epoch = 5  , validation_set=({'input': test_x}, {'targets': test_y}),
    snapshot_step=500, show_metric=True, run_id=Model_Name)

model.save(Model_Name)

import matplotlib.pyplot as plt
test_data = processTestData()
fig = plt.figure()
for num, data in enumerate(test_data[:1]):
    #cow [1,0]
    #alligator [0,1]
    img_num = data[1]
    img_data = data[0]
    y = fig.add_subplot(1,1,num+1)
    orig = img_data
    data = img_data.reshape(imageSize,imageSize,1)
    model_out = model.predict([data])
    str_label = ''
    if numpy.argmax(model_out) == 1: str_label='Tumor'
    else: str_label = 'Normal'

    y.imshow(orig, cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()