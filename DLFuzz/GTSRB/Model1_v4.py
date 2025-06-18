'''
Model 1 Versió 4: Modificant optimitzador
'''


# usage: python Model1_v4.py - train the model

from __future__ import print_function

from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense, Activation
from keras import regularizers
from keras.models import Model
from keras.utils import to_categorical
from utils_tmp import *
import scipy.misc
import os

from configs import bcolors

train_csv_path = '/content/gtsrb-german-traffic-sign/Train.csv'
test_csv_path = '/content/gtsrb-german-traffic-sign/Test.csv'

# Load the metadata into DataFrames
train_df = pd.read_csv(train_csv_path)



script_dir = os.path.dirname(os.path.abspath(__file__)) 
print(script_dir)


def Model1(input_tensor=None, train=False, retrain = False, newData = None, newTestData = None, inference_retrain = None):
    nb_classes = 43
    # convolution kernel size
    kernel_size = (5, 5)

    if train or retrain:
        batch_size = 256
        nb_epoch = 15

        # input image dimensions
        img_rows, img_cols = 64, 64
        
        if newData is not None and newTestData is not None:
            new_train_df = pd.read_csv(newData)
            test_df = pd.read_csv(newTestData)
            
            df_train = pd.concat([train_df, new_train_df], axis=0, ignore_index=True)
            df_train = df_train.sample(frac=1, random_state=18).reset_index(drop=True)
            
            x_train = np.array([load_and_preprocess_image(path, (img_rows, img_cols)) for path in df_train['Path']])
            y_train = to_categorical(df_train['ClassId'].values, nb_classes)
        
        else:
            x_train = np.array([load_and_preprocess_image(path, (img_rows, img_cols)) for path in train_df['Path']])
            y_train = to_categorical(train_df['ClassId'].values, nb_classes)

            test_df = pd.read_csv(test_csv_path)



        x_test = np.array([load_and_preprocess_image(path, (img_rows, img_cols)) for path in test_df['Path']])
        y_test = to_categorical(test_df['ClassId'].values, nb_classes)


        input_shape = (img_rows, img_cols, 3)

        print(x_train.shape)

        input_tensor = Input(shape=input_shape)



    elif input_tensor is None:
        print(bcolors.FAIL + 'you have to proved input_tensor when testing')
        exit()


    l2_lambda = 0.0001  

    # Block 1
    x = Conv2D(64, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_lambda))(input_tensor)
    x = Conv2D(128, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_lambda))(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)

    # Block 2
    x = Conv2D(256, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_lambda))(x)
    x = Conv2D(512, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_lambda))(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)

    # Flatten & Fully Connected Layers
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(43, name='before_softmax')(x)
    x = Activation('softmax', name='predictions')(x)


    model = Model(input_tensor, x)

    if train:
        # compiling
        model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

        # trainig
        model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=batch_size, epochs=nb_epoch, verbose=1)
        # save model
        model.save_weights('./Model1_v4.h5')
        score = model.evaluate(x_test, y_test, verbose=0)
        print('\n')
        print('Overall Test score:', score[0])
        print('Overall Test accuracy:', score[1])
    
    elif retrain:
        # Load previous model weights
        model.load_weights(os.path.join(script_dir, "Model1_v4.h5"))
        print(bcolors.OKBLUE + 'Model V4 weights loaded for retraining' + bcolors.ENDC)

        # Retrain with new data
        model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

        model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=batch_size, epochs=nb_epoch, verbose=1)

        # Save the updated weights after retraining
        model.save_weights('./Model1_v4_retrained.h5')
        print(bcolors.OKGREEN + 'Model V4 retrained and weights saved' + bcolors.ENDC)

    else:
        if inference_retrain is None:
          model.load_weights(os.path.join(script_dir, "Model1_v4.h5"))
          print(bcolors.OKBLUE + 'Model1 V4 loaded' + bcolors.ENDC)
        else:
          model.load_weights(os.path.join(script_dir, "Model1_v4_retrained.h5"))
          print(bcolors.OKBLUE + 'Model1 V4 retrained loaded' + bcolors.ENDC)

    return model


if __name__ == '__main__':
    Model1(train=True)
