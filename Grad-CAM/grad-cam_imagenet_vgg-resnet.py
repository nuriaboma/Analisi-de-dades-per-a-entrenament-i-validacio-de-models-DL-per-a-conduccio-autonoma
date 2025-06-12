from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
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
import os
import random

tf.set_random_seed(18)
random.seed(18)
np.random.seed(18)

def target_category_loss(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot([category_index], nb_classes))

def target_category_loss_output_shape(input_shape):
    return input_shape

def normalize(x):
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def load_image(path):
    img = image.load_img(path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def register_gradient():
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * tf.cast(op.inputs[0] > 0., dtype)

def compile_saliency_function(model, activation_layer):
    input_img = model.input
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
    layer_output = layer_dict[activation_layer].output
    max_output = K.max(layer_output, axis=3)
    saliency = K.gradients(K.sum(max_output), input_img)[0]
    return K.function([input_img, K.learning_phase()], [saliency])

def modify_backprop(model_name, name):
    g = tf.get_default_graph()
    with g.gradient_override_map({'Relu': name}):
        layer_dict = [layer for layer in model.layers[1:] if hasattr(layer, 'activation')]
        for layer in layer_dict:
            if layer.activation == keras.activations.relu:
                layer.activation = tf.nn.relu
        if model_name == 'vgg16':
            return VGG16(weights='imagenet')
        elif model_name == 'vgg19':
            return VGG19(weights='imagenet')
        elif model_name == 'resnet50':
            return ResNet50(weights='imagenet')
        else:
            raise ValueError("Unknown model for guided backprop")

def deprocess_image(x):
    if np.ndim(x) > 3:
        x = np.squeeze(x)
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    x += 0.5
    x = np.clip(x, 0, 1)
    x *= 255
    if K.image_dim_ordering() == 'th':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def grad_cam(input_model, image, category_index, layer_name):
    # Troba la capa convolucional
    conv_layer = None
    for layer in input_model.layers:
        if layer.name == layer_name:
            conv_layer = layer
            break
    if conv_layer is None:
        raise ValueError("Could not find layer with name: " + layer_name)

    nb_classes = 1000
    target = target_category_loss(input_model.output, category_index, nb_classes)
    loss = K.sum(target)
    #conv_output = conv_layer.output
    conv_output = model.get_layer(layer_name).output
    grads = normalize(K.gradients(loss, conv_output)[0])
    gradient_function = K.function([model.input, K.learning_phase()], [conv_output, grads])

    output, grads_val = gradient_function([image, 0])  # 0 per a inference
    output, grads_val = output[0], grads_val[0]

    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.ones(output.shape[0:2], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    cam = cv2.resize(cam, (224, 224))
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)

    image = image[0]
    image -= np.min(image)
    image = np.minimum(image, 255)

    cam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    cam = np.float32(cam) + np.float32(image)
    cam = 255 * cam / np.max(cam)
    return np.uint8(cam), heatmap

# --- Main ---

image_path = sys.argv[1]
model_name = sys.argv[2].lower()
preprocessed_input = load_image(image_path)

if model_name == 'vgg16':
    model = VGG16(weights='imagenet')
    last_conv_layer = 'block5_conv3'
elif model_name == 'vgg19':
    model = VGG19(weights='imagenet')
    last_conv_layer = 'block5_conv4'
elif model_name == 'resnet50':
    model = ResNet50(weights='imagenet')
    last_conv_layer = 'activation_49'
else:
    raise ValueError("No Model Specified")

#last_conv_layer = None
#for layer in model.layers[::-1]:
#    if len(layer.output_shape) == 4: 
#        last_conv_layer = layer.name
#        break
#print("Using last convolutional layer:", last_conv_layer)


predictions = model.predict(preprocessed_input)
top_1 = decode_predictions(predictions)[0][0]
print('Predicted class:')
print('%s (%s) with probability %.2f' % (top_1[1], top_1[0], top_1[2]))

predicted_class = np.argmax(predictions)
cam, heatmap = grad_cam(model, preprocessed_input, predicted_class, last_conv_layer)

register_gradient()
guided_model = modify_backprop(model_name, 'GuidedBackProp')
if model_name == 'resnet50':
    model = ResNet50(weights='imagenet')
    last_conv_layer = 'activation_98'
saliency_fn = compile_saliency_function(guided_model, last_conv_layer)
saliency = saliency_fn([preprocessed_input, 0])
gradcam = saliency[0] * heatmap[..., np.newaxis]



if not os.path.exists('./Grad_CAM_imagenet'):
    os.makedirs('./Grad_CAM_imagenet')

filename = os.path.basename(image_path)
filename, _ = os.path.splitext(filename)
conf = "{:.3f}".format(top_1[2])

output_path = './Grad_CAM_imagenet/gradcam__' + filename + '__@' + top_1[1] + '@__' + str(conf) + '__' + model_name + '.jpg'
output_path_guided = './Grad_CAM_imagenet/guided_gradcam__' + filename + '__@' + top_1[1] + '@__' + str(conf) + '__' + model_name + '.jpg'

cv2.imwrite(output_path, cam)
cv2.imwrite(output_path_guided, deprocess_image(gradcam))
