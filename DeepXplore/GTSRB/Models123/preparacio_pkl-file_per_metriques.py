import numpy as np
import pickle
from keras.models import Model
import tensorflow as tf

from utils import load_and_preprocess_image
import pandas as pd
from keras.utils import to_categorical

from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense, Activation
from keras import regularizers

def create_model1():
    input_shape = (64, 64, 3)
    input_tensor = Input(shape=input_shape)

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
    return model


def create_model2():
    input_shape = (64, 64, 3)
    input_tensor = Input(shape=input_shape)

    x = Conv2D(filters=32, kernel_size=(5, 5), activation='relu')(input_tensor)
    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(rate=0.25)(x)

    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(rate=0.25)(x)

    # Flatten and Fully Connected Layers
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(rate=0.5)(x)
    x = Dense(43, name='before_softmax')(x)
    x = Activation('softmax', name='predictions')(x)

    model = Model(input_tensor, x)
    return model


def create_model3():
    input_shape = (64, 64, 3)
    input_tensor = Input(shape=input_shape)

    # block1
    x = Conv2D(filters=32, kernel_size=(5, 5), activation='relu')(input_tensor)
    x = Conv2D(filters=64, kernel_size=(5, 5), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(rate=0.15)(x)

    x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(rate=0.20)(x)

    # Flatten and Fully Connected Layers
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(rate=0.25)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(rate=0.25)(x)
    x = Dense(43,name='before_softmax')(x)
    x = Activation('softmax', name='predictions')(x)


    model = Model(input_tensor, x)
    return model


model1 = create_model1()
model1.load_weights('TFG/DeepXplore/GTSRB/Models_123/Model1.h5')
# model1.load_weights('TFG/DeepXplore/GTSRB/Models_123/Model1_retrained.h5')
model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model2 = create_model2()
model2.load_weights('TFG/DeepXplore/GTSRB/Models_123/Model2.h5')
# model2.load_weights('TFG/DeepXplore/GTSRB/Models_123/Model2_retrained.h5')
model2.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

model3 = create_model3()
model3.load_weights('TFG/DeepXplore/GTSRB/Models_123/Model3.h5')
# model3.load_weights('TFG/DeepXplore/GTSRB/Models_123/Model3_retrained.h5')
model3.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])


# test_csv_path = '/content/gtsrb-german-traffic-sign/Test.csv'
test_csv_path = 'TFG/DeepXplore/GTSRB/Models_123/Model_tot_test_df.csv'
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
with open("/content/predictions_modelsbons_before_retrain.pkl", "wb") as f:
    pickle.dump((y_probs1, y_probs2, y_probs3, y_pred1, y_pred2, y_pred3, x_test, y_test), f)

print("Predictions saved successfully.")
