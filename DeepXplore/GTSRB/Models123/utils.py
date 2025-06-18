import random
from collections import defaultdict

import numpy as np
from keras import backend as K
from keras.models import Model
from keras.preprocessing import image
from configs import *
import pandas as pd
import tensorflow as tf
from scipy.stats import wasserstein_distance
from PIL import Image



def load_and_preprocess_image(path, img_size):
    # Load the image
    if path.startswith('Train') or path.startswith('Test'):
        img = image.load_img('/content/gtsrb-german-traffic-sign/{}'.format(path))
    else:
      img = image.load_img(path)
    # Resize to target size (Ensure images are consistent in size)
    img = img.resize(img_size, Image.BICUBIC)

    # Convert to numpy array and normalize
    img_array = np.array(img, dtype=np.float32) / 255.0

    # Ensure the shape is (height, width, channels)
    if len(img_array.shape) == 2:  # Grayscale image (H, W)
        img_array = np.stack([img_array] * 3, axis=-1)  # Convert to (H, W, 3)

    return img_array

def deprocess_image(x):
    x = x * 255.0  # Reverse normalization (scale back to [0, 255])
    x = np.clip(x, 0, 255).astype('uint8')  # Ensure values are within valid range
    x = x.reshape((64, 64, 3))  # Reshape to original image dimensions
    return x



def decode_label(pred):
    return gtsrb_labels[pred]


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def constraint_occl(gradients, start_point, rect_shape):
    new_grads = np.zeros_like(gradients)
    new_grads[:, start_point[0]:start_point[0] + rect_shape[0],
    start_point[1]:start_point[1] + rect_shape[1]] = gradients[:, start_point[0]:start_point[0] + rect_shape[0],
                                                     start_point[1]:start_point[1] + rect_shape[1]]
    return new_grads


def constraint_light(gradients):
    new_grads = np.ones_like(gradients)
    grad_mean = 1e4 * np.mean(gradients)
    return grad_mean * new_grads


def constraint_black(gradients, rect_shape=(10, 10)):
    start_point = (
        random.randint(0, gradients.shape[1] - rect_shape[0]), random.randint(0, gradients.shape[2] - rect_shape[1]))
    new_grads = np.zeros_like(gradients)
    patch = gradients[:, start_point[0]:start_point[0] + rect_shape[0], start_point[1]:start_point[1] + rect_shape[1]]
    if np.mean(patch) < 0:
        new_grads[:, start_point[0]:start_point[0] + rect_shape[0],
        start_point[1]:start_point[1] + rect_shape[1]] = -np.ones_like(patch)
    return new_grads


def init_coverage_tables(model1, model2, model3):
    model_layer_dict1 = defaultdict(bool)
    model_layer_dict2 = defaultdict(bool)
    model_layer_dict3 = defaultdict(bool)
    init_dict(model1, model_layer_dict1)
    init_dict(model2, model_layer_dict2)
    init_dict(model3, model_layer_dict3)
    return model_layer_dict1, model_layer_dict2, model_layer_dict3


def init_dict(model, model_layer_dict):
    for layer in model.layers:
        if 'flatten' in layer.name or 'input' in layer.name:
            continue
        for index in range(layer.output_shape[-1]):
            model_layer_dict[(layer.name, index)] = False


def neuron_to_cover(model_layer_dict):
    not_covered = [(layer_name, index) for (layer_name, index), v in model_layer_dict.items() if not v]
    if not_covered:
        layer_name, index = random.choice(not_covered)
    else:
        layer_name, index = random.choice(model_layer_dict.keys())
    return layer_name, index


def neuron_covered(model_layer_dict):
    covered_neurons = len([v for v in model_layer_dict.values() if v])
    total_neurons = len(model_layer_dict)
    return covered_neurons, total_neurons, covered_neurons / float(total_neurons)


def scale(intermediate_layer_output, rmax=1, rmin=0):
    X_std = (intermediate_layer_output - intermediate_layer_output.min()) / (
            intermediate_layer_output.max() - intermediate_layer_output.min())
    X_scaled = X_std * (rmax - rmin) + rmin
    return X_scaled


def update_coverage(input_data, model, model_layer_dict, threshold=0):
    layer_names = [layer.name for layer in model.layers if
                   'flatten' not in layer.name and 'input' not in layer.name]

    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=[model.get_layer(layer_name).output for layer_name in layer_names])
    intermediate_layer_outputs = intermediate_layer_model.predict(input_data)

    for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
        scaled = scale(intermediate_layer_output[0])
        for num_neuron in xrange(scaled.shape[-1]):
            if np.mean(scaled[..., num_neuron]) > threshold and not model_layer_dict[(layer_names[i], num_neuron)]:
                model_layer_dict[(layer_names[i], num_neuron)] = True


def full_coverage(model_layer_dict):
    if False in model_layer_dict.values():
        return False
    return True


def fired(model, layer_name, index, input_data, threshold=0):
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    intermediate_layer_output = intermediate_layer_model.predict(input_data)[0]
    scaled = scale(intermediate_layer_output)
    if np.mean(scaled[..., index]) > threshold:
        return True
    return False


def diverged(predictions1, predictions2, predictions3, target):
    #     if predictions2 == predictions3 == target and predictions1 != target:
    if not predictions1 == predictions2 == predictions3:
        return True
    return False



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

def get_top_predictions(pred_before, pred, model_num):
    top_10_before = np.argsort(pred_before.flatten())[-10:][::-1]
    top_10_probs_before = pred_before.flatten()[top_10_before]
    top_10_before = top_10_before.tolist()
    top_10_probs_before = top_10_probs_before.tolist()
    
    
    top_10_after = np.argsort(pred.flatten())[-10:][::-1]
    top_10_probs_after = pred.flatten()[top_10_after]
    top_10_after = top_10_after.tolist()
    top_10_probs_after = top_10_probs_after.tolist()

    print_final = {'Labels_before': top_10_before, 'Probs_before': top_10_probs_before, 'Labels_after': top_10_after, 'Probs_after': top_10_probs_after}
    print(bcolors.DARKGREEN + 'Model {} - Predictions: {}'.format(
            model_num, print_final) + bcolors.ENDC)


def get_10_labels(pred):
    top_10 = np.argsort(pred.flatten())[-5:][::-1] 
    top_10_probs = pred.flatten()[top_10] 

    top_10 = top_10.tolist() 
    top_10_probs = top_10_probs.tolist()
    top_10_labels = [decode_label(idx) for idx in top_10]

    print(bcolors.DARKBLUE + 'Top 5 Predictions: ' + bcolors.ENDC)
    for label, prob in zip(top_10_labels, top_10_probs):
        print(bcolors.OKBLUE + "{} ({:.4f})".format(label, prob) + bcolors.ENDC)

