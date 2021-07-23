from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization, Activation

# set image generators
train_dir = '/home/oms315/Desktop/HW3/chest_xray/train/'
test_dir = '/home/oms315/Desktop/HW3/chest_xray/test/'
validation_dir = '/home/oms315/Desktop/HW3/chest_xray/val/'

train_datagen = ImageDataGenerator(rescale = 1. / 255)
test_datagen = ImageDataGenerator(rescale = 1. / 255)
validation_datagen = ImageDataGenerator(rescale = 1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size = (512, 512),
    batch_size = 10,
    class_mode = 'binary')
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size = (512, 512),
    batch_size = 10,
    class_mode = 'binary')
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size = (512, 512),
    batch_size = 10,
    class_mode = 'binary')

for data_batch, labels_batch in train_generator:
    print('data_size:', data_batch.shape)
    print('labels_size:', labels_batch.shape)
    break

# model definition


input_shape = [512, 512, 3]  # as a shape of image


def build_model():
    model = models.Sequential()
    conv_base = VGG16(weights = 'imagenet',
                      include_top = False,
                      input_shape = input_shape)
    conv_base.trainable = False

    model.add(conv_base)

    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(layers.Dense(128, activation = 'relu'))
    model.add(layers.Dense(1, activation = 'sigmoid'))

    # compile
    model.compile(optimizer = optimizers.RMSprop(lr = 1e-5),
                  loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model


# main loop without cross-validation
import time

starttime = time.time();
num_epochs = 100
model = build_model()
history = model.fit_generator(train_generator,
                              epochs = num_epochs, steps_per_epoch = 100,
                              validation_data = validation_generator, validation_steps = 50)

# saving the model
model.save('hw3_model.h5')

# evaluation
train_loss, train_acc = model.evaluate_generator(train_generator)
test_loss, test_acc = model.evaluate_generator(test_generator)
print('train_loss:', train_loss)
print('train_acc:', train_acc)
print('test_loss:', test_loss)
print('test_acc:', test_acc)
print("elapsed time (in sec): ", time.time() - starttime)