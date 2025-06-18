'''
usage: python gen_diff.py -h
'''

from __future__ import print_function

import argparse

from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.layers import Input
import scipy.misc
import os

from configs import bcolors
from utils import *

random.seed(18)
np.random.seed(18)

# read the parameter
# argument parsing
parser = argparse.ArgumentParser(
    description='Main function for difference-inducing input generation in ImageNet dataset')
parser.add_argument('transformation', help="realistic transformation type", choices=['light', 'occl', 'blackout'])
parser.add_argument('weight_diff', help="weight hyperparm to control differential behavior", type=float)
parser.add_argument('weight_nc', help="weight hyperparm to control neuron coverage", type=float)
parser.add_argument('step', help="step size of gradient descent", type=float)
parser.add_argument('seeds', help="number of seeds of input", type=int)
parser.add_argument('grad_iterations', help="number of iterations of gradient descent", type=int)
parser.add_argument('threshold', help="threshold for determining neuron activated", type=float)
parser.add_argument('-t', '--target_model', help="target model that we want it predicts differently",
                    choices=[0, 1, 2], default=0, type=int)
parser.add_argument('-sp', '--start_point', help="occlusion upper left corner coordinate (e.g., '10,10')", default="0,0", type=str)
parser.add_argument('-occl_size', '--occlusion_size', help="occlusion size (e.g., '10,10')", default="50,50", type=str)

args = parser.parse_args()

args.start_point = tuple(map(int, args.start_point.split(',')))
args.occlusion_size = tuple(map(int, args.occlusion_size.split(',')))

# input image dimensions
img_rows, img_cols = 224, 224
input_shape = (img_rows, img_cols, 3)

# define input tensor as a placeholder
input_tensor = Input(shape=input_shape)

# load multiple models sharing same input tensor
K.set_learning_phase(0)
model1 = VGG16(input_tensor=input_tensor)
model2 = VGG19(input_tensor=input_tensor)
model3 = ResNet50(input_tensor=input_tensor)
# init coverage table
model_layer_dict1, model_layer_dict2, model_layer_dict3 = init_coverage_tables(model1, model2, model3)
print(args)

img_dir = './seeds_20'
img_pathss = os.listdir(img_dir)
img_num = len(img_pathss)


# ==============================================================================================

# start
img_paths = image.list_pictures('./seeds_20/', ext='JPEG')
#for i in xrange(args.seeds):
    #gen_img = preprocess_image(random.choice(img_paths))
    #rand_index = random.randint(0, len(img_paths) - 1)
for i in xrange(img_num):
    iimmgg_path = img_paths[i]
    
    print('Image path: ',iimmgg_path)
    
    gen_img = preprocess_image(img_paths[i])

    orig_img = gen_img.copy()
    # first check if input already induces differences
    pred1, pred2, pred3 = model1.predict(gen_img), model2.predict(gen_img), model3.predict(gen_img)
    label1, label2, label3 = np.argmax(pred1[0]), np.argmax(pred2[0]), np.argmax(pred3[0])

    if not label1 == label2 == label3:
        print(bcolors.OKGREEN + 'input already causes different outputs: {}, {}, {}'.format(decode_label(pred1),
                                                                                            decode_label(pred2),
                                                                                            decode_label(
                                                                                                pred3)) + bcolors.ENDC)

        update_coverage(gen_img, model1, model_layer_dict1, args.threshold)
        update_coverage(gen_img, model2, model_layer_dict2, args.threshold)
        update_coverage(gen_img, model3, model_layer_dict3, args.threshold)

        print(bcolors.OKGREEN + 'covered neurons percentage %d neurons %.3f, %d neurons %.3f, %d neurons %.3f'
              % (len(model_layer_dict1), neuron_covered(model_layer_dict1)[2], len(model_layer_dict2),
                 neuron_covered(model_layer_dict2)[2], len(model_layer_dict3),
                 neuron_covered(model_layer_dict3)[2]) + bcolors.ENDC)
        averaged_nc = (neuron_covered(model_layer_dict1)[0] + neuron_covered(model_layer_dict2)[0] +
                       neuron_covered(model_layer_dict3)[0]) / float(
            neuron_covered(model_layer_dict1)[1] + neuron_covered(model_layer_dict2)[1] +
            neuron_covered(model_layer_dict3)[
                1])
        print(bcolors.OKGREEN + 'averaged covered neurons %.3f' % averaged_nc + bcolors.ENDC)

        gen_img_deprocessed = deprocess_image(gen_img)

        # save the result to disk
        scipy.misc.imsave('./generated_inputs/' + 'already_differ_' + decode_label(pred1) + '_' + decode_label(
            pred2) + '_' + decode_label(pred3) + '_' + str(i) + '.png', gen_img_deprocessed)

        print('./generated_inputs/' + 'already_differ_' + decode_label(pred1) + '_' + decode_label(pred2) + '_' + decode_label(pred3) + '_' + str(i) + '.png')

        continue


    # if all label agrees
    pred1_before = pred1[:]  
    pred2_before = pred2[:] 
    pred3_before = pred3[:] 


    orig_label = label1
    layer_name1, index1 = neuron_to_cover(model_layer_dict1)
    layer_name2, index2 = neuron_to_cover(model_layer_dict2)
    layer_name3, index3 = neuron_to_cover(model_layer_dict3)

    # construct joint loss function
    if args.target_model == 0:
        loss1 = -args.weight_diff * K.mean(model1.get_layer('predictions').output[..., orig_label])
        loss2 = K.mean(model2.get_layer('predictions').output[..., orig_label])
        loss3 = K.mean(model3.get_layer('fc1000').output[..., orig_label])
    elif args.target_model == 1:
        loss1 = K.mean(model1.get_layer('predictions').output[..., orig_label])
        loss2 = -args.weight_diff * K.mean(model2.get_layer('predictions').output[..., orig_label])
        loss3 = K.mean(model3.get_layer('fc1000').output[..., orig_label])
    elif args.target_model == 2:
        loss1 = K.mean(model1.get_layer('predictions').output[..., label1])
        loss2 = K.mean(model2.get_layer('predictions').output[..., orig_label])
        loss3 = -args.weight_diff * K.mean(model3.get_layer('fc1000').output[..., orig_label])
    loss1_neuron = K.mean(model1.get_layer(layer_name1).output[..., index1])
    loss2_neuron = K.mean(model2.get_layer(layer_name2).output[..., index2])
    loss3_neuron = K.mean(model3.get_layer(layer_name3).output[..., index3])
    layer_output = (loss1 + loss2 + loss3) + args.weight_nc * (loss1_neuron + loss2_neuron + loss3_neuron)

    # for adversarial image generation
    final_loss = K.mean(layer_output)

    # we compute the gradient of the input picture wrt this loss
    grads = normalize(K.gradients(final_loss, input_tensor)[0])

    # this function returns the loss and grads given the input picture
    iterate = K.function([input_tensor], [loss1, loss2, loss3, loss1_neuron, loss2_neuron, loss3_neuron, grads])

    # we run gradient ascent for 20 steps
    for iters in xrange(args.grad_iterations):
        loss_value1, loss_value2, loss_value3, loss_neuron1, loss_neuron2, loss_neuron3, grads_value = iterate(
            [gen_img])
        if args.transformation == 'light':
            grads_value = constraint_light(grads_value)  # constraint the gradients value
        elif args.transformation == 'occl':
            grads_value = constraint_occl(grads_value, args.start_point,
                                          args.occlusion_size)  # constraint the gradients value
        elif args.transformation == 'blackout':
            grads_value = constraint_black(grads_value)  # constraint the gradients value

        gen_img += grads_value * args.step
        pred1, pred2, pred3 = model1.predict(gen_img), model2.predict(gen_img), model3.predict(gen_img)
        label1, label2, label3 = np.argmax(pred1[0]), np.argmax(pred2[0]), np.argmax(pred3[0])

        if not label1 == label2 == label3:
            print(bcolors.OKBLUE + 'Models predictions before: Model 1 -> {} ({}), Model 2 -> {} ({}), Model 3 -> {} ({})'.format(
                    decode_label(pred1_before), np.max(pred1_before[0]),
                    decode_label(pred2_before), np.max(pred2_before[0]),
                    decode_label(pred3_before), np.max(pred3_before[0])
                ) + bcolors.ENDC)
            
            print(bcolors.OKBLUE + 'Models predictions after: '
                'Model 1 -> {} (Conf: {:.4f}), (Original predicion conf: {:.4f}), '
                'Model 2 -> {} (Conf: {:.4f}), (Original predicion conf: {:.4f}), '
                'Model 3 -> {} (Conf: {:.4f}), (Original predicion conf: {:.4f})'.format(
                    decode_label(pred1), np.max(pred1[0]), pred1[0][orig_label],  
                    decode_label(pred2), np.max(pred2[0]), pred2[0][orig_label],  
                    decode_label(pred3), np.max(pred3[0]), pred3[0][orig_label]  
                ) + bcolors.ENDC)
            
            #Print 10 highest predictions for each model
            for j, (pred_before, pred) in enumerate([(pred1_before, pred1), 
                                         (pred2_before, pred2), 
                                         (pred3_before, pred3)], start=1):
                get_top_predictions(pred_before, pred, j)
            
            print(bcolors.DARKYELLOW + 'EMD Score: Model 1 -> {}, Model 2 -> {}, Model 3 -> {}'.format(
                    wasserstein_distance(range(len(pred1_before[0])), range(len(pred1[0])), pred1_before[0], pred1[0]),
                    wasserstein_distance(range(len(pred2_before[0])), range(len(pred2[0])), pred2_before[0], pred2[0]),
                    wasserstein_distance(range(len(pred3_before[0])), range(len(pred3[0])), pred3_before[0], pred3[0])
                ) + bcolors.ENDC)
            
            
            update_coverage(gen_img, model1, model_layer_dict1, args.threshold)
            update_coverage(gen_img, model2, model_layer_dict2, args.threshold)
            update_coverage(gen_img, model3, model_layer_dict3, args.threshold)

            print(bcolors.OKGREEN + 'covered neurons percentage %d neurons %.3f, %d neurons %.3f, %d neurons %.3f'
                  % (len(model_layer_dict1), neuron_covered(model_layer_dict1)[2], len(model_layer_dict2),
                     neuron_covered(model_layer_dict2)[2], len(model_layer_dict3),
                     neuron_covered(model_layer_dict3)[2]) + bcolors.ENDC)
            averaged_nc = (neuron_covered(model_layer_dict1)[0] + neuron_covered(model_layer_dict2)[0] +
                           neuron_covered(model_layer_dict3)[0]) / float(
                neuron_covered(model_layer_dict1)[1] + neuron_covered(model_layer_dict2)[1] +
                neuron_covered(model_layer_dict3)[
                    1])
            print(bcolors.OKGREEN + 'averaged covered neurons %.3f' % averaged_nc + bcolors.ENDC)

            gen_img_deprocessed = deprocess_image(gen_img)
            orig_img_deprocessed = deprocess_image(orig_img)

            # save the result to disk
            scipy.misc.imsave(
                './generated_inputs/' + args.transformation + '_' + decode_label(pred1) + '_' + decode_label(
                    pred2) + '_' + decode_label(pred3) + '_' + str(i) + '.png', gen_img_deprocessed)
            scipy.misc.imsave(
                './generated_inputs/' + args.transformation + '_' + decode_label(pred1) + '_' + decode_label(
                    pred2) + '_' + decode_label(pred3) + '_' + str(i) + '_orig.png', orig_img_deprocessed)
            
            print('./generated_inputs/' + args.transformation + '_' + decode_label(pred1) + '_' + decode_label(pred2) + '_' + decode_label(pred3) + '_' + str(i) + '.png')
            
            break
