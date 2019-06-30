from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling1D, Conv1D, UpSampling2D, MaxPooling1D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import wave
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
import argparse

from tqdm import tqdm

class BatchGenerator():
    def __init__(self, batch_size, directory):
        self.MAX_SIZE = int.from_bytes(b"\xFF\xFF", "little")
        
        self.batch_size = batch_size
        self.directory = directory

        self.data = self._load_data()


    def fetch_random_batch(self):
        np.random.shuffle(self.data)

        return self.data[0:self.batch_size]
    
    def _convert_bytes_to_tensor(self, byte_list):
        four_wide = [(byte_list[i*4:i*4+2], byte_list[i*4+2:i*4+4]) for i in range(len(byte_list) // 4)]

        channel1_list = [int.from_bytes(four_wide[i][0], "little") / self.MAX_SIZE for i in range(len(four_wide))]
        channel2_list = [int.from_bytes(four_wide[i][1], "little") / self.MAX_SIZE for i in range(len(four_wide))]
        
        signal_tensor = np.column_stack([channel1_list, channel2_list])

        return signal_tensor
    
    def convert_tensor_to_bytes(self, tensor):
        tensor *= self.MAX_SIZE
        tensor = tensor.astype(np.int16)

        byte_list = tensor.tobytes()

        return byte_list

    def _load_data(self):
        data = []

        first = True
        for file in tqdm(os.listdir(self.directory)):
            if file.endswith(".wav"):
                wave_reader = wave.open(self.directory + file, "rb")

                if first:
                    self.channels = wave_reader.getnchannels()
                    self.sampwidth = wave_reader.getsampwidth()
                    self.framerate = wave_reader.getframerate()
                    self.frames_per_sample = wave_reader.getnframes()
                    first = False

                byte_list = wave_reader.readframes(wave_reader.getnframes())
                sample_tensor = self._convert_bytes_to_tensor(byte_list)

                data.append(sample_tensor)
        
        return np.asarray(data)
    
    def save_bytes_to_wav(self, byte_list, filename):
        wave_writer = wave.open(filename, "wb")

        wave_writer.setnchannels(self.channels)
        wave_writer.setsampwidth(self.sampwidth)
        wave_writer.setframerate(self.framerate)
        wave_writer.writeframes(byte_list)

        wave_writer.close()

class DCGAN():
    def __init__(self, inputdir, outputdir, batch_size, interval, modelfile=None):
        # Set up batch generator
        self.bg = BatchGenerator(batch_size, inputdir)
        self.batch_size = batch_size
        self.inputdir = inputdir
        self.outputdir = outputdir
        self.interval = interval
        self.modelfile = modelfile

        # Input shape
        self.sample_frames = 441000
        self.channels = 2
        self.sample_shape = (self.sample_frames, self.channels)
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])
        print("Discriminator built")
        # Build the generator
        self.generator = self.build_generator()
        print("Generator built")
        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)
        

    def build_generator(self):

        model = Sequential()
        model.add(Dense(1 * 11025, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((11025, 1)))
        model.add(Conv1D(256, kernel_size=7, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Reshape((11025, 1, 256)))
        model.add(UpSampling2D(size=(5, 1)))
        model.add(Reshape((55125, 256)))
        model.add(Conv1D(128, kernel_size=7, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Reshape((55125, 1, 128)))
        model.add(UpSampling2D(size=(2, 1)))
        model.add(Reshape((110250, 128)))
        model.add(Conv1D(128, kernel_size=7, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Reshape((110250, 1, 128)))
        model.add(UpSampling2D(size=(2, 1)))
        model.add(Reshape((220500, 128)))
        model.add(Conv1D(64, kernel_size=7, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Reshape((220500, 1, 64)))
        model.add(UpSampling2D(size=(2, 1)))
        model.add(Reshape((441000, 64)))
        model.add(Conv1D(32, kernel_size=7, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv1D(self.channels, kernel_size=7, padding="same"))
        model.add(Activation("sigmoid"))
        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv1D(32, kernel_size=3, strides=2, input_shape=self.sample_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv1D(64, kernel_size=3, strides=2, padding="same"))
        #model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv1D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv1D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(MaxPooling1D(pool_size=2, padding="valid"))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=self.sample_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs):

        #self.bg.data = np.expand_dims(self.bg.data, axis=3)

        # Adversarial ground truths
        valid = np.ones((self.batch_size, 1))
        fake = np.zeros((self.batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------


            batch = self.bg.fetch_random_batch()

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (self.batch_size, self.latent_dim))
            gen_samples = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(batch, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_samples, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % self.interval == 0:
                self.save_samples(epoch)

    def save_samples(self, epoch):
        n = 5
        noise = np.random.normal(0, 1, (n, self.latent_dim))
        gen_samples = self.generator.predict(noise)

        for i in range(n):
            tensor = gen_samples[i]
            byte_list = self.bg.convert_tensor_to_bytes(tensor)
            path = self.outputdir + "sample{}".format(epoch) + "_{}.wav".format(i)
            self.bg.save_bytes_to_wav(byte_list, path)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("inputdir", help="Directory containing audio samples")
    parser.add_argument("outputdir", help="Directory to output samples during training")
    parser.add_argument("--batchsize", type=int, default=8, help="Batch size used during training")
    parser.add_argument("--interval", type=int, default=10, help="Interval between saving samples during training")
    parser.add_argument("--modelfile", help="File to save model at after training")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")

    args = parser.parse_args()

    gan = DCGAN(args.inputdir, args.outputdir, args.batchsize, args.interval, args.modelfile)
    gan.train(args.epochs)

if __name__ == '__main__':
    main()



