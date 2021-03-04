# ======================================================================
# There are 5 questions in this exam with increasing difficulty from 1-5.
# Please note that the weight of the grade for the question is relative
# to its difficulty. So your Category 1 question will score significantly
# less than your Category 5 question.
#
# Don't use lambda layers in your model.
# You do not need them to solve the question.
# Lambda layers are not supported by the grading infrastructure.
#
# You must use the Submit and Test button to submit your model
# at least once in this category before you finally submit your exam,
# otherwise you will score zero for this category.
# ======================================================================
#
# Computer Vision with CNNs
#
# Build a classifier for Rock-Paper-Scissors based on the rock_paper_scissors
# TensorFlow dataset.
#
# IMPORTANT: Your final layer should be as shown. Do not change the
# provided code, or the tests may fail
#
# IMPORTANT: Images will be tested as 150x150 with 3 bytes of color depth
# So ensure that your input layer is designed accordingly, or the tests
# may fail. 
#
# NOTE THAT THIS IS UNLABELLED DATA. 
# You can use the ImageDataGenerator to automatically label it
# and we have provided some starter code.


import urllib.request
import zipfile
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense,Conv2D,Dropout,Flatten,MaxPooling2D
from tensorflow.keras.models import Sequential,load_model


def solution_model():
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/rps.zip'
    urllib.request.urlretrieve(url, 'C:/data/tf_test3/rps.zip')
    local_zip = 'C:/data/tf_test3/rps.zip'
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall('C:/data/tf_test3/rps.zip')
    zip_ref.close()


    TRAINING_DIR = "tmp/rps/"
    training_datagen = ImageDataGenerator(
        width_shift_range =0.1,
        height_shift_range = 0.1,
        validation_split=0.2,
        resize=1/255
    )
    train_generator = training_datagen.flow_from_directory(    
        "C:/data/tf_test3/rps.zip",
        target_size =(150, 150) ,
        batch_size = 32,
        class_mode = 'train'
        )

    test_generator = training_datagen.flow_from_directory(    
        "C:/data/tf_test3/rps.zip",
        target_size =(150, 150) ,
        batch_size = 32,
        class_mode = 'valdation'
        )

    model = tf.keras.models.Sequential([
    # YOUR CODE HERE, BUT END WITH A 3 Neuron Dense, activated by softmax
    model.add(Conv2D(filters=400, kernel_size=(2,2), padding = 'same', strides=2
                        ,input_shape=(150,150,3)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Dense(32))
    model.add(Dense(16))
    model.add(Dense(8))
    model.add(Dense(3,activation='softmax'))
    ])
    return model

    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['acc'])
    model.fit(train_generator, test_generator, epochs=100, verbose=1 )


# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("mymodel.h5")
