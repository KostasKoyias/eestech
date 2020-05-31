import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Flatten, Dropout
from keras.layers import BatchNormalization, Conv2D, MaxPooling2D
from keras.models import Sequential, Model
from keras import optimizers
from keras_preprocessing.image import ImageDataGenerator

DATAPATH = "/home/kostas/Desktop/coding/contests/kaggle/eestech/input/2020-athens-eestech-challenge"

def naive_predict(traindf: pd.DataFrame, testdf: pd.DataFrame):
    datagen=ImageDataGenerator(rescale=1./255.,validation_split=0.15)
    train_generator=datagen.flow_from_dataframe(
        dataframe=traindf,
        directory= DATAPATH + "train/",
        x_col="file",
        y_col="category",
        subset="training",
        batch_size=32,
        shuffle=True,
        class_mode="categorical",
        target_size=(64,64))

    valid_generator=datagen.flow_from_dataframe(
        dataframe=traindf,
        directory=DATAPATH + "train/",
        x_col="file",
        y_col="category",
        subset="validation",
        batch_size=32,
        shuffle=True,
        class_mode="categorical",
        target_size=(64,64))

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                    input_shape=(64,64,3)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(31, activation='softmax'))
    model.compile(optimizers.rmsprop(lr=0.0005, decay=1e-6),loss="categorical_crossentropy",metrics=["accuracy"])
    model.summary()

    STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
    STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
    model.fit_generator(generator=train_generator,
                        steps_per_epoch=STEP_SIZE_TRAIN,
                        validation_data=valid_generator,
                        validation_steps=STEP_SIZE_VALID,
                        epochs=5
    )
    model.evaluate_generator(generator=valid_generator, steps=STEP_SIZE_VALID
    )

    test_datagen=ImageDataGenerator(rescale=1./255.)
    test_generator=test_datagen.flow_from_dataframe(
        dataframe=testdf,
        directory=DATAPATH + "test/",
        x_col="file",
        y_col=None,
        batch_size=32,
        shuffle=False,
        class_mode=None,
        target_size=(64,64))
    test_generator.reset()
    pred=model.predict_generator(test_generator, verbose=1)
    predicted_class_indices=np.argmax(pred,axis=1)

    labels = (train_generator.class_indices)
    labels = dict((v,k) for k,v in labels.items())
    predictions = [labels[k] for k in predicted_class_indices]
    predictions.insert(0, 'category')
    my_indexes = list(range(1, len(predictions)+1))
    my_indexes = [int(i) for i in my_indexes] 
    my_indexes.insert(0, 'id')
    results = zip(my_indexes, predictions)

    results = pd.DataFrame(results)
    results.to_csv('results.csv')