from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model, Sequential
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img

from keras.layers import Dense, Activation, Flatten
from keras.layers import Dense, Conv2D, Dropout, BatchNormalization, Input, Reshape, Flatten, Deconvolution2D, Conv2DTranspose, MaxPooling2D, UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img
from keras.optimizers import adam
#########################################
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras import backend as K
from keras import optimizers
import warnings
warnings.filterwarnings('ignore')


import numpy as np


import keras
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, MaxPool2D, Flatten, BatchNormalization
from keras.layers import Conv1D, MaxPool1D, CuDNNLSTM, Reshape
from keras.layers import Input, Dense, Dropout, Activation, Add, Concatenate
from keras.datasets import cifar10
from keras import regularizers
from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import SGD, Adam, RMSprop, Adadelta
import keras.backend as K
from keras.objectives import mean_squared_error
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils

from sklearn.utils import class_weight
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer, RobustScaler, StandardScaler


import load_data
from sklearn import preprocessing


# return DataSet class
data = load_data.read_data_sets(one_hot=True)

# get train data and labels
x_train, y_train = data.train.next_batch(84420)

# get test data
x_test = data.test.data

# get test labels
y_test = data.test.labels


import numpy as np

def transfor(dataset, w):
    im = np.zeros([9, 9])
    im[0, 3] = dataset[0]
    im[0, 4] = dataset[1]
    im[0, 5] = dataset[2]
    
    im[1, 3] = dataset[3]
    im[1, 5] = dataset[4]

    im[2, 0] = dataset[5]
    im[2, 1] = dataset[6]
    im[2, 2] = dataset[7]
    im[2, 3] = dataset[8]
    im[2, 4] = dataset[9]
    im[2, 5] = dataset[10]
    im[2, 6] = dataset[11]
    im[2, 7] = dataset[12]
    im[2, 8] = dataset[13]

    im[3, 0] = dataset[14] * w ##
    im[3, 1] = dataset[15]
    im[3, 2] = dataset[16]
    im[3, 3] = dataset[17]
    im[3, 4] = dataset[18]
    im[3, 5] = dataset[19]
    im[3, 6] = dataset[20]
    im[3, 7] = dataset[21]
    im[3, 8] = dataset[22] * w ##

    im[4, 0] = dataset[23] * w ##
    im[4, 1] = dataset[24] * w ##
    im[4, 2] = dataset[25]
    im[4, 3] = dataset[26]
    im[4, 4] = dataset[27]
    im[4, 5] = dataset[28]
    im[4, 6] = dataset[29]
    im[4, 7] = dataset[30] * w ##
    im[4, 8] = dataset[31] * w ##

    im[5, 0] = dataset[32] * w ##
    im[5, 1] = dataset[33] * w ##
    im[5, 2] = dataset[34]
    im[5, 3] = dataset[35]
    im[5, 4] = dataset[36]
    im[5, 5] = dataset[37]
    im[5, 6] = dataset[38]
    im[5, 7] = dataset[39] * w ##
    im[5, 8] = dataset[40] * w ##

    im[6, 0] = dataset[41] * w ##
    im[6, 1] = dataset[42]
    im[6, 2] = dataset[43]
    im[6, 3] = dataset[44]
    im[6, 4] = dataset[45]
    im[6, 5] = dataset[46]
    im[6, 6] = dataset[47]
    im[6, 7] = dataset[48]
    im[6, 8] = dataset[49] * w ##

    im[7, 1] = dataset[50]
    im[7, 2] = dataset[51]
    im[7, 3] = dataset[52]
    im[7, 4] = dataset[53]
    im[7, 5] = dataset[54]
    im[7, 6] = dataset[55]
    im[7, 7] = dataset[56]

    im[8, 2] = dataset[57]
    im[8, 3] = dataset[58]
    im[8, 4] = dataset[59]
    im[8, 5] = dataset[60]
    im[8, 6] = dataset[61]

    im = im.reshape(im.shape[0], im.shape[1], 1)
    return im
        
def convert(d,l):
    images = []
    labels = []
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    d = min_max_scaler.fit_transform(d.T).transpose()
    for i in range(d.shape[0]):
        label = l[i]
        image = convert2img(d[i])
        images.append(image)

        labels.append(label)
    return np.array(images), np.array(labels)
     
def convert2img(datarow):
    w = 1.2
    f0 = transfor(datarow[:62], w)
    f1 = transfor(datarow[62:124], w)
    f2 = transfor(datarow[124:186], w)
    f3 = transfor(datarow[186:248], w)
    f4 = transfor(datarow[248:310], w)
    
    image = np.concatenate((f0, f1, f2, f3, f4), axis=2)
    return image

x_trainImg, y_trainImg = convert(x_train, y_train)

x_testImg, y_testImg = convert(x_test, y_test)

def create_block(input, channel): ## Convolution block of 2 layers
    x = input
    for i in range(2):
        x = Conv2D(channel, 3, padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
    return x


batch_size = 1024
epochs=50




input = Input((9,9,5))
    
    # Encoder
block1 = create_block(input, 32)
x = MaxPool2D(2)(block1)
block2 = create_block(x, 64)
x = MaxPool2D(2)(block2)
    
middle = create_block(x, 128)
    
    # Decoder
up1 = UpSampling2D((3,3))(middle)
block3 = create_block(up1, 64)
    #up1 = UpSampling2D((2,2))(block3)
up2 = UpSampling2D((3,3))(block3)
block4 = create_block(up2, 32)
up2 = MaxPool2D(2)(block4)
    #up2 = create_block(up2,32)
    # output
x = Conv2D(5, 1)(up2)
output = Activation("sigmoid")(x)

encoder =  Model(input, middle)
model = Model(input, output)

model.compile(Adam(0.001, 0.9), loss='categorical_crossentropy',  metrics=['accuracy'])
model.summary()


historyAutoencoder = model.fit(x_trainImg, x_trainImg, 
                       batch_size=512,
                       epochs=100,
                       verbose=1,
                       validation_data=(x_testImg, x_testImg),
                       shuffle=True)






encoder.summary()


predict_train = encoder.predict(x_trainImg)
predict_test = encoder.predict(x_testImg)



input = Input((predict_train.shape[1], predict_train.shape[2], predict_train.shape[3]))
x = Conv2D(1024, 3, padding="same")(input)
x = Activation('relu')(x)
x = BatchNormalization()(x)
x = MaxPool2D(2)(x)
x = Dropout(0.5)(x)
    # x = Conv2D(128, 3, padding="same")(x)
x = Activation('relu')(x)
x = BatchNormalization()(x)
    # x = MaxPool2D(2)(x)
x = Dropout(0.5)(x)
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.35)(x)
x = Dense(100, activation='relu')(x)
x = Dropout(0.69)(x)
output = Dense(3, activation='softmax')(x)

decoder = Model(input, output)



decoder.compile(loss='categorical_crossentropy',
                        optimizer=Adam(),
                        metrics=['accuracy'])

decoder.summary()


train_datagen = ImageDataGenerator(shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
train_set = train_datagen.flow(predict_train, y_trainImg, batch_size=512)

test_datagen = ImageDataGenerator()
test_set = test_datagen.flow(predict_test,y_testImg, batch_size=512)


early = EarlyStopping(monitor='val_acc', patience=10, restore_best_weights=True)
learning = ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=5, min_delta=0.0001)
callbacks = [early, learning]


AutoencoderConvClassifierImageGenerator = decoder.fit_generator(train_set,
                        epochs=100,
                        steps_per_epoch=np.ceil(x_trainImg.shape[0]/512),
                        verbose=1,
                        validation_data=(test_set),
                        validation_steps=np.ceil(x_testImg.shape[0]/512))

size = np.ceil(x_testImg.shape[0]/512)

decoder.evaluate_generator(test_set,steps=size)


decoder.evaluate(predict_test,y_testImg)

decoder.save('model_78.44.h5')


decoder_ae_conv.save('model_77.h5')



############################################################
############################################################