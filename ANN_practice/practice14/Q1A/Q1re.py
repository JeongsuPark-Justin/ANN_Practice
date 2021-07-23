from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10, mnist
import numpy as np
import matplotlib.pyplot as plt


def Conv2D(filters, kernel_size, padding='same', activation='relu'):
    return layers.Conv2D(filters, kernel_size, padding=padding, activation=activation)


class AE(models.Model):
    def __init__(self, org_shape=(1, 32, 32, 3)):

        original = layers.Input(shape=org_shape)

        x = Conv2D(4, (3, 3))(original)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(8, (3, 3))(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        z = Conv2D(1, (8, 8))(x)

        y = Conv2D(16, (3, 3))(z)
        y = layers.UpSampling2D((2, 2))(y)

        y = Conv2D(8, (3, 3))(y)
        y = layers.UpSampling2D((2, 2))(y)

        y = Conv2D(4, (3, 3))(y)

        decoded = Conv2D(3, (3, 3), activation='sigmoid')(y)

        super().__init__(original, decoded)
        self.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

        self.original = original
        self.z = z

        self.summary()

    def Encoder(self):
        return models.Model(self.original, self.z)

    def Decoder(self):
        z_shape = (self.z_dim[1], )
        z = layers.Input(shape=z_shape)
        h = self.layers[-6][z]
        h = self.layers[-5][h]
        h = self.layers[-4][h]
        h = self.layers[-3][h]
        h = self.layers[-2][h]
        h = self.layers[-1][h]
        return models.Model(z, h)


def data_load():
    (X_train, _), (X_test, _) = cifar10.load_data()

    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255
    # [_, W, H] = X_train.shape
    # X_train = X_train.reshape((-1, W, H, 3))
    # X_test = X_test.reshape((-1, W, H, 3))
    return (X_train, X_test)


def plot_acc(h, title="accuracy"):
    plt.plot(h.history['accuracy'])
    plt.plot(h.history['val_accuracy'])
    plt.title(title)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc=0)


def plot_loss(h, title="loss"):
    plt.plot(h.history['loss'])
    plt.plot(h.history['val_loss'])
    plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc=0)


def show_ae(autoencoder, X_test):
    encoder = autoencoder.Encoder()

    encoded_imgs = encoder.predict(X_test)
    decoded_imgs = autoencoder.predict(X_test)

    n = 10
    plt.figure(figsize=(20, 6))
    for i in range(n):

        ax = plt.subplot(4, n, i + 1)
        plt.imshow(X_test[i].reshape(32, 32, 3))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(4, n, i + 1 + n)
        plt.stem(encoded_imgs[i].reshape(-1), use_line_collection=True)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(4, n, i + 1 + n + n)
        plt.imshow(encoded_imgs[i].reshape(8, 8, 1))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(4, n, i + 1 + n + n + n)
        plt.imshow(decoded_imgs[i].reshape(32, 32, 3))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)


input_shape = [32, 32, 3]

(X_train, X_test) = data_load()
autoencoder = AE(input_shape)

history = autoencoder.fit(X_train, X_train, epochs=20, batch_size=128, shuffle=True,
                          validation_data=(X_test, X_test))

plot_loss(history)
plt.savefig('AE_LOSS.png')
plt.clf()
plot_acc(history)
plt.savefig('AE_ACC.png')
show_ae(autoencoder, X_test)
plt.savefig('AE_predicted.png')
plt.show()
