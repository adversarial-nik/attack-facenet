import numpy as np
from keras.models import load_model
import cv2
from imageio import imread
from skimage.transform import resize
from scipy.spatial import distance

input_image_path = 'images/mark.jpg'
target_image_path = 'images/bill/Bill_Gates_0002.jpg'
output_image_path = 'results/mark_to_bill.png'

model_path = 'model/keras/model/facenet_keras.h5'
cascade_path = 'model/cv2/haarcascade_frontalface_alt2.xml'
image_size = 160


def prewhiten(image):
    # map values from [from_min, from_max] to [to_min, to_max]
    # image: input array
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

# load model
model = load_model(model_path)

# load images
target_image = prewhiten(load_and_align_images([target_image_path]))
ip_image = prewhiten(load_and_align_images([input_image_path]))
hacked_image = prewhiten(load_and_align_images([output_image_path]))

# check validate attack
target_emb = model.predict(target_image)
input_emb = model.predict(ip_image)
hacked_emb = model.predict(hacked_image)

print("distance of target with input image: ", distance.euclidean(target_emb, input_emb))
print("distance of target with hacked image: ", distance.euclidean(target_emb, hacked_emb))
