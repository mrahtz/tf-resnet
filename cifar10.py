import numpy as np
from sacred import Experiment
from sacred.observers import FileStorageObserver
from tensorflow.python.keras.callbacks import ReduceLROnPlateau, TensorBoard
from tensorflow.python.keras.datasets import cifar10
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from resnet import make_resnet
from utils import tf_disable_warnings, tf_disable_deprecation_warnings

tf_disable_warnings()
tf_disable_deprecation_warnings()

ex = Experiment()
observer = FileStorageObserver.create('runs')
ex.observers.append(observer)

ex.add_config({
    'net': 'resnet20'
})


@ex.automain
def main(net):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train.astype('float32') / 255, x_test.astype('float32') / 255
    mean = np.mean(x_train, axis=0)
    x_train -= mean
    x_test -= mean

    datagen = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
    )
    datagen.fit(x_train)

    model = make_resnet(net)
    model.summary()

    model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
                        validation_data=(x_test, y_test),
                        epochs=200,
                        callbacks=[ReduceLROnPlateau(verbose=1),
                                   TensorBoard(observer.dir)])
