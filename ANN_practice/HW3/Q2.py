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
    target_size = (128, 128),
    batch_size = 20,
    class_mode = 'binary')
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size = (128, 128),
    batch_size = 20,
    class_mode = 'binary')
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size = (128, 128),
    batch_size = 20,
    class_mode = 'binary')

for data_batch, labels_batch in train_generator:
    print('data_size:', data_batch.shape)
    print('labels_size:', labels_batch.shape)
    break

# model definition


from tensorflow.keras.models import load_model

model = load_model('hw3_model.h5')
conv_base = model.layers[0]

for layer in conv_base.layers:
    if layer.name.startswith('block5'):
        layer.trainable = True
model.compile(optimizer = optimizers.RMSprop(lr = 1e-5),
              loss = 'binary_crossentropy', metrics = ['accuracy'])

# main loop without cross-validation
import time

starttime = time.time()
num_epochs = 100
history = model.fit_generator(train_generator,
                              epochs = num_epochs, steps_per_epoch = 100,
                              validation_data = validation_generator, validation_steps = 50)

model.save('hw3_fine_model.h5')

# evaluation
train_loss, train_acc = model.evaluate_generator(train_generator)
test_loss, test_acc = model.evaluate(test_generator)
print('train_fine_loss:', train_loss)
print('train_fine_acc:', train_acc)
print('test_fine_loss:', test_loss)
print('test_fine_acc:', test_acc)
print("elapsed time (in sec): ", time.time() - starttime)


# visualization
def plot_acc(h, title="accuracy"):
    plt.plot(h.history['acc'])
    plt.plot(h.history['val_acc'])
    plt.title(title)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc = 0)


def plot_loss(h, title='loss'):
    plt.plot(h.history['loss'])
    plt.plot(h.history['val_loss'])
    plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc = 0)


plot_loss(history)
plt.savefig('loss1.png')
plt.clf()
plot_acc(history)
plt.savefig('acc1.png')
