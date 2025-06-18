'''
usage: python gen_diff.py -h
'''

from __future__ import print_function

import argparse

from Model1 import Model1
from Model2 import Model2
from Model3 import Model3
from keras.layers import Input
import scipy.misc
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import cv2

from configs import bcolors
from utils import *
import os

random.seed(18)
np.random.seed(18)

parser = argparse.ArgumentParser()
parser.add_argument('weights', help="weights used", choices=['trained', 'retrained'])
parser.add_argument('images', help="images you want to use", choices=['country', 'type'])
args = parser.parse_args()

def show_image(img_path, idx):
    folder_name = args.weights + args.images
    folder_path = os.path.join('/content', folder_name)
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        # print(f"Created directory: {folder_path}")
    
    img = cv2.imread(img_path)
    if img is None:
        print("Image not found:", img_path)
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.axis('off')
    
    output_path = os.path.join(folder_path, 'tmp_output_{}.png'.format(idx))
    plt.savefig(output_path, bbox_inches='tight')
    print("SHOW_IMAGE: {}".format(output_path))


# input image dimensions
img_rows, img_cols = 64, 64
input_shape = (img_rows, img_cols, 3)

if args.images == 'country':
    img_dir = '/content/altres_senyals/altres_paisos'
else:
    img_dir = '/content/altres_senyals/altres_tipus'

img_paths = os.listdir(img_dir)
img_num = len(img_paths)

# define input tensor as a placeholder
input_tensor = Input(shape=input_shape)

# load multiple models sharing same input tensor
if args.weights == 'retrained':
  model1 = Model1(input_tensor=input_tensor, inference_retrain = True)
  model2 = Model2(input_tensor=input_tensor, inference_retrain = True)
  model3 = Model3(input_tensor=input_tensor, inference_retrain = True)
else:
  model1 = Model1(input_tensor=input_tensor)
  model2 = Model2(input_tensor=input_tensor)
  model3 = Model3(input_tensor=input_tensor)


# ==============================================================================================
for idx, i in enumerate(xrange(img_num)):
    img_path = os.path.join(img_dir,img_paths[i])

    img_name = img_paths[i].split('.')[0]

    print('Image path: ',img_path)
    show_image(img_path, idx)

    iimmgg = load_and_preprocess_image(img_path, (img_rows, img_cols))
    gen_img = np.expand_dims(iimmgg, axis=0)

    orig_img = gen_img.copy()
    # first check if input already induces differences
    pred1, pred2, pred3 = model1.predict(gen_img), model2.predict(gen_img), model3.predict(gen_img)
    label1, label2, label3 = np.argmax(pred1[0]), np.argmax(pred2[0]), np.argmax(pred3[0])

    # print(bcolors.OKGREEN + 'Model 1: {}, Model 2: {}, Model 3: {}'.format(decode_label(label1),
                                                                                            # decode_label(label2),
                                                                                            # decode_label(label3)) + bcolors.ENDC)

    print(bcolors.OKGREEN + 'Models predictions:'+ bcolors.ENDC)
    print(bcolors.DARKGREEN + 'Model 1 -> {} ({})'.format(
                      decode_label(label1), np.max(pred1[0]),
                  ) + bcolors.ENDC)
    get_10_labels(pred1)
    print('\n')
    
    print(bcolors.DARKGREEN + 'Model 2 -> {} ({})'.format(
                      decode_label(label2), np.max(pred2[0]),
                  ) + bcolors.ENDC)
    get_10_labels(pred2)
    print('\n')

    print(bcolors.DARKGREEN + 'Model 3 -> {} ({})'.format(

                      decode_label(label3), np.max(pred3[0])
                  ) + bcolors.ENDC)
    get_10_labels(pred3)
    print('\n')

