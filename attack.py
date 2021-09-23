import numpy as np
from keras.preprocessing import image
from keras.applications import inception_v3
from tensorflow.keras import backend as K
from PIL import Image
import tensorflow as tf
from keras.models import load_model
from keras.models import Model
import cv2
from imageio import imread
from skimage.transform import resize
from scipy.spatial import distance
from tensorflow.keras.losses import MeanSquaredError
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


tf.compat.v1.disable_eager_execution()

def dewhiten(image):
    # convert image pixel values to range 0-255
    to_min = 0
    to_max = 255
    from_min = np.min(image)
    from_max = np.max(image)
    from_range = from_max - from_min
    to_range = to_max - to_min
    scaled = np.array((image - from_min) / float(from_range), dtype=float)
    return to_min + (scaled * to_range)

def prewhiten(image):
    # convert image pixel values to range 0-1
    to_min = 0
    to_max = 1
    from_min = np.min(image)
    from_max = np.max(image)
    from_range = from_max - from_min
    to_range = to_max - to_min
    scaled = np.array((image - from_min) / float(from_range), dtype=float)
    return to_min + (scaled * to_range)

def load_and_align_images(filepaths, margin=1):
    cascade = cv2.CascadeClassifier(cascade_path)
    
    aligned_images = []
    for filepath in filepaths:
        img = imread(filepath)

        faces = cascade.detectMultiScale(img,
                                         scaleFactor=1.1,
                                         minNeighbors=3)
        (x, y, w, h) = faces[0]
        cropped = img[y-margin//2:y+h+margin//2,
                      x-margin//2:x+w+margin//2, :]
        aligned = resize(cropped, (image_size, image_size), mode='reflect')
        aligned_images.append(aligned)
            
    return np.array(aligned_images)


input_image_path = 'images/mark.jpg'

target_image_path = 'images/bill/Bill_Gates_0002.jpg'
output_image_path = 'hacked_image.png'

model_path = 'model/keras/model/facenet_keras.h5'
cascade_path = 'model/cv2/haarcascade_frontalface_alt2.xml'
image_size = 160

# load model
##########################################################
print("[INFO] loading pre-trained facenet model...")
model = load_model(model_path)


ip_layer = model.layers[0].input
op_layer = model.layers[-1].output
##########################################################

# Load the image to hack
# load target image
ip_image = prewhiten(load_and_align_images([input_image_path]))
target_image = prewhiten(load_and_align_images([target_image_path]))


# set bounds for change during optimization
max_change_above = ip_image + 0.01
max_change_below = ip_image - 0.01


# Create a copy of the input image to hack on
hacked_image = np.copy(ip_image)
print(hacked_image.shape, type(hacked_image))

# get prediction for target from model
target_pred = model.predict(target_image)[0]

# define cost
mse = MeanSquaredError()
cost_function = mse(tf.convert_to_tensor(target_pred), op_layer) 

gradient_function = K.gradients(cost_function, ip_layer)[0]
grab_cost_and_gradients_from_model = K.function([ip_layer, K.learning_phase()], [cost_function, gradient_function])
cost = 0.0
learning_rate = 0.9



for epoch in range(300):

    cost, gradients = grab_cost_and_gradients_from_model([hacked_image, 0])
    
    # apply gradients
    hacked_image -= gradients * learning_rate
    hacked_image = np.clip(hacked_image, 0.0, 1.0)

    print("\033[92m [+] cost: {:.8}\033[0m".format(cost))

hacked_img = hacked_image[0] 
im = image.array_to_img(dewhiten(hacked_img))
im.save(output_image_path)


diff_img = hacked_image[0] - ip_image[0]
diff_im = image.array_to_img(dewhiten(diff_img))
diff_im.save('/tmp/diff.png')


# check validate attack
target_emb = model.predict(target_image)
input_emb = model.predict(ip_image)
hacked_emb = model.predict(hacked_image)

print("distance of target with input image: ", distance.euclidean(target_emb, input_emb))
print("distance of target with hacked image: ", distance.euclidean(target_emb, hacked_emb))
