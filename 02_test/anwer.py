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
# Basic Datasets Question
#
# Create a classifier for the Fashion MNIST dataset
# Note that the test will expect it to classify 10 classes and that the
# input shape should be the native size of the Fashion MNIST dataset which is
# 28x28 monochrome. Do not resize the data. Your input layer should accept
# (28,28) as the input shape only. If you amend this, the tests will fail.
#
import tensorflow as tf
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Conv1D, Flatten, Dense, MaxPooling1D,Dropout
from tensorflow.keras.datasets import fashion_mnist

def solution_model():
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


    # YOUR CODE HERE
    model = Sequential()
    model.add(Conv1D(filters=64,input_shape=(28,28) ,activation='relu',kernel_size=(3),strides=1,padding='valid'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Dense(32))
    model.add(Dense(10, activation='softmax'))
    return model

    model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['acc'])
    model.fit(x_train, y_train, epochs=100, validation_size=0.2, verbose=1,batch_size=32)

# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("C:/study/tf_certificate/Category2/mymodel.h5")
    model = load_model("C:/study/tf_certificate/Category2/mymodel.h5")

loss, accuracy = model.evaluate(x_test,y_test, batch_size=64)
print("loss,accuracy:",loss,acc)
y_pred = model.predict(x_test)

