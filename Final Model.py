from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np


categories = ['no', 'yes']

model = load_model('FinalProject_FT_ep200.h5')

test_dir = '/home/boamike/PycharmProjects/pythonProject/project/test/testest'

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(400, 400),
    batch_size=30,
    class_mode='binary')

pre = model.predict_generator(test_generator)
eval = model.evaluate_generator(test_generator)
# np.set_printoptions(formatter={'float': lambda x: '{0:0:3f}'.format(x)})

print(test_generator.class_indices)
# print('pre')
# print('%s: %.2f%%' %(model.metrics_names[1], pre[1]*100))
print('eval')
print('%s: %.2f%%' %(model.metrics_names[1], eval[1]*100))



