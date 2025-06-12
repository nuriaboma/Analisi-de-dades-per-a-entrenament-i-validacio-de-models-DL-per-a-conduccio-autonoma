from keras.preprocessing import image
from keras.layers.core import Lambda
from keras.models import Sequential
from tensorflow.python.framework import ops
import keras.backend as K
import tensorflow as tf
import numpy as np
import keras
import sys
import cv2
import sys
from PIL import Image
import cv2
import os
import re 
from keras.layers import Input, Conv2D
from keras.models import Model
import random

from Model1 import Model1
from Model2 import Model2
from Model3 import Model3 

tf.set_random_seed(18)
random.seed(18)
np.random.seed(18)

def load_and_preprocess_image(path, img_size = (64, 64)):
    # Load the image
    if path.startswith('Train') or path.startswith('Test'):
        img = image.load_img('/content/gtsrb-german-traffic-sign/{}'.format(path))
    else:
      img = image.load_img(path)
    # Resize to target size 
    img = img.resize(img_size, Image.BICUBIC)

    # Convert to numpy array and normalize
    img_array = np.array(img, dtype=np.float32) / 255.0

    # Ensure the shape is (height, width, channels)
    if len(img_array.shape) == 2:  # Grayscale image (H, W)
        img_array = np.stack([img_array] * 3, axis=-1)  # Convert to (H, W, 3)

    exp_img = np.expand_dims(img_array, axis=0)

    return exp_img

def decode_predictions(pred):
    return gtsrb_labels[pred]

gtsrb_labels = {
    0: "Speed limit_20 km-h",
    1: "Speed limit_30 km-h",
    2: "Speed limit_50 km-h",
    3: "Speed limit_60 km-h",
    4: "Speed limit_70 km-h",
    5: "Speed limit_80 km-h",
    6: "End of speed limit_80 km-h",
    7: "Speed limit_100 km-h",
    8: "Speed limit_120 km-h",
    9: "No passing",
    10: "No passing for vehicles over 3.5 metric tons",
    11: "Right-of-way at the next intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5 metric tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve to the left",
    20: "Dangerous curve to the right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice-snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing for vehicles over 3.5 metric tons"
}

def target_category_loss(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot([category_index], nb_classes))

def target_category_loss_output_shape(input_shape):
    return input_shape

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def load_image(path):
    return load_and_preprocess_image(path, img_size=(64, 64)) 

def register_gradient():
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * \
                tf.cast(op.inputs[0] > 0., dtype)

def compile_saliency_function(model, activation_layer):
    input_img = model.input
    layer_dict = dict((layer.name, layer) for layer in model.layers)
    layer_output = layer_dict[activation_layer].output
    max_output = K.max(layer_output, axis=3)
    saliency = K.gradients(K.sum(max_output), input_img)[0]
    return K.function([input_img, K.learning_phase()], [saliency])

def modify_backprop(model, name):
    g = tf.get_default_graph()
    with g.gradient_override_map({'Relu': name}):

        # get layers that have an activation
        layer_dict = [layer for layer in model.layers[1:]
                      if hasattr(layer, 'activation')]

        # replace relu activation
        for layer in layer_dict:
            if layer.activation == keras.activations.relu:
                layer.activation = tf.nn.relu

        # re-instanciate a new model
        if model_name == 'model1':
            new_model = Model1(input_tensor=input_tensor)
        elif model_name == 'model2':
            new_model = Model2(input_tensor=input_tensor)
        elif model_name == 'model3':
            new_model = Model3(input_tensor=input_tensor)
        else:
            raise ValueError("Unknown model for guided backprop")
    return new_model

def deprocess_image(x):
    x = np.array(x)
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    x += 0.5
    x = np.clip(x, 0, 1)
    x *= 255
    return np.uint8(x)


def grad_cam(input_model, image, category_index, layer_name):
    model = Sequential()
    model.add(input_model)

    nb_classes = model.output_shape[-1]
    target_layer = lambda x: target_category_loss(x, category_index, nb_classes)
    model.add(Lambda(target_layer, output_shape=target_category_loss_output_shape))

    loss = K.sum(model.layers[-1].output)
    
    conv_output = [l for l in model.layers[0].layers if l.name == layer_name][0].output
    grads = normalize(K.gradients(loss, conv_output)[0])
    
    gradient_function = K.function([model.input, K.learning_phase()], [conv_output, grads])
    output, grads_val = gradient_function([image, 0]) 
    output, grads_val = output[0], grads_val[0]

    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.ones(output.shape[0:2], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    cam = cv2.resize(cam, (image.shape[2], image.shape[1]))
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)

    image = image[0]
    image -= np.min(image)
    image = np.minimum(image, 255)

    cam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    cam = np.float32(cam) + np.float32(image * 255)
    cam = 255 * cam / np.max(cam)
    return np.uint8(cam), heatmap




image_path = sys.argv[1]
model_name = sys.argv[2].lower()

preprocessed_input = load_image(sys.argv[1])

img_rows, img_cols = 64, 64
input_shape = (img_rows, img_cols, 3)
input_tensor = Input(shape=input_shape)

if model_name == 'model1':
    model = Model1(input_tensor=input_tensor)
    last_conv_layer = 'conv2d_4'
elif model_name == 'model2':
    model = Model2(input_tensor=input_tensor)
    last_conv_layer = 'conv2d_3'
elif model_name == 'model3':
    model = Model3(input_tensor=input_tensor)
    last_conv_layer = 'conv2d_4'
else:
    raise ValueError("No Model Specified")

predictions = model.predict(preprocessed_input)
predicted_class = np.argmax(predictions)
print('Predicted class:')
print('%s with probability %.2f' % (decode_predictions(np.argmax(predictions[0])), np.max(predictions[0])))

#for layer in model.layers:
#    print(layer.name, layer.output_shape)


predicted_class = np.argmax(predictions)
cam, heatmap = grad_cam(model, preprocessed_input, predicted_class, last_conv_layer)

if model_name == 'model2':
    last_conv_layer = 'conv2d_6'
else:
    last_conv_layer = 'conv2d_8'

register_gradient()
guided_model = modify_backprop(model, 'GuidedBackProp')
saliency_fn = compile_saliency_function(guided_model, last_conv_layer)
saliency = saliency_fn([preprocessed_input, 0])[0]
saliency = saliency[0]  
saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-5)
gradcam = saliency * heatmap[..., np.newaxis]
guided_gradcam_img = deprocess_image(gradcam)
guided_gradcam_img_bgr = cv2.cvtColor(guided_gradcam_img, cv2.COLOR_RGB2BGR)


if not os.path.exists('./Grad_CAM_GTSRB'):
    os.makedirs('./Grad_CAM_GTSRB')

filename = os.path.basename(image_path)
filename, _ = os.path.splitext(filename)
conf = "{:.3f}".format(np.max(predictions[0]))
output_path = '/content/Grad_CAM_GTSRB/gradcam__' + filename + '__@' + decode_predictions(np.argmax(predictions[0])) + '@__' + str(conf) + '__' + model_name + '.jpg'
output_path_guided = '/content/Grad_CAM_GTSRB/guided_gradcam__' + filename + '__@' + decode_predictions(np.argmax(predictions[0])) + '@__' + str(conf) + '__' + model_name + '.jpg'

cv2.imwrite(output_path, cam)
cv2.imwrite(output_path_guided, guided_gradcam_img_bgr)
