import numpy as np
import pickle
from keras.models import load_model, Model
import tensorflow as tf

from utils import load_and_preprocess_image
import pandas as pd
from keras.utils import to_categorical

from keras.layers import Input, Convolution2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense, Activation
from keras import regularizers

nb_classes = 43
kernel_size = (5, 5)

def create_model1():
    input_shape = (64, 64, 3)
    input_tensor = Input(shape=input_shape)

    # block1
    x = Convolution2D(4, kernel_size, activation='relu', padding='same', name='block1_conv1')(input_tensor)
    x = MaxPooling2D(pool_size=(2, 2), name='block1_pool1')(x)

    # block2
    x = Convolution2D(12, kernel_size, activation='relu', padding='same', name='block2_conv1')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='block2_pool1')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(nb_classes, name='before_softmax')(x)
    x = Activation('softmax', name='predictions')(x)


    model = Model(input_tensor, x)
    return model


def create_model2():
    input_shape = (64, 64, 3)
    input_tensor = Input(shape=input_shape)

    # block1
    x = Convolution2D(6, kernel_size, activation='relu', padding='same', name='block1_conv1')(input_tensor)
    x = MaxPooling2D(pool_size=(2, 2), name='block1_pool1')(x)

    # block2
    x = Convolution2D(16, kernel_size, activation='relu', padding='same', name='block2_conv1')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='block2_pool1')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(84, activation='relu', name='fc1')(x)
    x = Dense(nb_classes, name='before_softmax')(x)
    x = Activation('softmax', name='predictions')(x)

    model = Model(input_tensor, x)
    return model


def create_model3():
    input_shape = (64, 64, 3)
    input_tensor = Input(shape=input_shape)

    # block1
    x = Convolution2D(6, kernel_size, activation='relu', padding='same', name='block1_conv1')(input_tensor)
    x = MaxPooling2D(pool_size=(2, 2), name='block1_pool1')(x)

    # block2
    x = Convolution2D(16, kernel_size, activation='relu', padding='same', name='block2_conv1')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='block2_pool1')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(120, activation='relu', name='fc1')(x)
    x = Dense(84, activation='relu', name='fc2')(x)
    x = Dense(nb_classes, name='before_softmax')(x)
    x = Activation('softmax', name='predictions')(x)


    model = Model(input_tensor, x)
    return model


model1 = create_model1()
model1.load_weights('/content/drive/MyDrive/GIA/TFG/Senyals_deepXplore/img_retrain/Model1.h5')
model1.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

model2 = create_model2()
model2.load_weights('/content/drive/MyDrive/GIA/TFG/Senyals_deepXplore/img_retrain/Model2.h5')
model2.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

model3 = create_model3()
model3.load_weights('/content/drive/MyDrive/GIA/TFG/Senyals_deepXplore/img_retrain/Model3.h5')
model3.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])


# model1 = create_model1()
# model1.load_weights('/content/drive/MyDrive/GIA/TFG/Senyals_deepXplore/img_retrain/Model1_retrained.h5')
# model1.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

# model2 = create_model2()
# model2.load_weights('/content/drive/MyDrive/GIA/TFG/Senyals_deepXplore/img_retrain/Model2_retrained.h5')
# model2.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

# model3 = create_model3()
# model3.load_weights('/content/drive/MyDrive/GIA/TFG/Senyals_deepXplore/img_retrain/Model3_retrained.h5')
# model3.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])


# test_csv_path = '/content/gtsrb-german-traffic-sign/Test.csv'
test_csv_path = '/content/drive/MyDrive/GIA/TFG/Senyals_deepXplore/img_retrain/Model_tot_test_df.csv'
test_df = pd.read_csv(test_csv_path)

x_test = np.array([load_and_preprocess_image(path, (64, 64)) for path in test_df['Path']])
y_test = to_categorical(test_df['ClassId'].values, 43)


y_probs1 = model1.predict(x_test)
y_probs2 = model2.predict(x_test)
y_probs3 = model3.predict(x_test)

y_pred1 = np.argmax(y_probs1, axis=1)
y_pred2 = np.argmax(y_probs2, axis=1)
y_pred3 = np.argmax(y_probs3, axis=1)

# Save results to a file
with open("/content/drive/MyDrive/GIA/TFG/Senyals_deepXplore/img_retrain/predictions_before_retrain.pkl", "wb") as f:
    pickle.dump((y_probs1, y_probs2, y_probs3, y_pred1, y_pred2, y_pred3, x_test, y_test), f)

print("Predictions saved successfully.")
