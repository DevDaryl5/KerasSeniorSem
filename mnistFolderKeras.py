import sys
import numpy as np
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix


# define cnn model
def define_model():
    """
    Edit this function for most of the hyperperamiters 
    """
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(28, 28, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    # a dense layer is a layer that is deeply connected with its preceding layer
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    # Binary classification. sigmoid. binary_crossentropy. Dog vs cat, Sentiemnt analysis(pos/neg). 
    # https://www.dlology.com/blog/how-to-choose-last-layer-activation-and-loss-function/
    model.add(Dense(1, activation='sigmoid'))
    # compile model. Momentum is known to speed up learning and to help not getting stuck in local minima
    opt = SGD(learning_rate=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# plot diagnostic learning curves
def summarize_diagnostics(history):
    # plot loss
    pyplot.subplot(211)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='orange', label='test')
    # plot accuracy
    pyplot.subplot(212)
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['accuracy'], color='blue', label='train')
    pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
    pyplot.show()
    # save plot to file
    filename = sys.argv[0].split('/')[-1]
    pyplot.savefig(filename + '_plot.png')
    pyplot.close()

# run the test harness for evaluating a model
def run_test_harness():
    # define model
    model = define_model()
    # create data generator. Scale the pixel values to the range of 0-1
    datagen = ImageDataGenerator(rescale=1.0/255.0)
    # prepare iterators
    train_it = datagen.flow_from_directory('dataset/train/',
        class_mode='binary', batch_size=64, target_size=(28, 28))
    test_it = datagen.flow_from_directory('dataset/test/',
        class_mode='binary', batch_size=64, target_size=(28, 28))
    # fit model
    #EDIT EPOCHS HERE
    history = model.fit(train_it, steps_per_epoch=len(train_it),
        validation_data=test_it, validation_steps=len(test_it), epochs=10, verbose=2)
    # evaluate model
    _, acc = model.evaluate(test_it, steps=len(test_it), verbose=2)
    print('> %.3f' % (acc * 100.0))
    
    #Confution Matrix and Classification Report
    Y_pred = model.predict(test_it, 10000 // 65)
    print(Y_pred)
    y_pred = []
    for val in Y_pred:
        if val > 0.5:
            y_pred.append(1)
        else:
            y_pred.append(0)
    # y_pred = np.argmax(Y_pred, axis=1)
    print(y_pred)
    print('Confusion Matrix')
    print(confusion_matrix(test_it.classes, y_pred))
    print('Classification Report')
    target_names = ['Marked', 'Unmarked']
    print(classification_report(test_it.classes, y_pred, target_names=target_names))
    # learning curves
    summarize_diagnostics(history)


run_test_harness()


