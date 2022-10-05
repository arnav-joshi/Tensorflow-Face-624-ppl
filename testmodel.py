from operator import indexOf
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
import PIL
import PIL.Image
import tensorflow_datasets as tfds
import pathlib

mod2 = tf.keras.models.load_model('model_saved')

# Check architecture
mod2.summary()

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "put test image here")
DIR = pathlib.Path(DIR)
img_height = 180
img_width = 180
test_ds = tf.keras.utils.image_dataset_from_directory(
    DIR,
    seed=123,
    image_size=(img_height, img_width))

pred = mod2.predict(test_ds)
print(pred.shape)
out = pred.tolist()
people = ["Eric","Jessie","Kartik","Kayla","Peyton","Sid"]
for img in out:
    print("image " + str(out.index(img)))
    high = 0.0
    for person in img:
        if(person>high):
            highindex = img.index(person)
            high = person
    print("This is " + people[highindex])




