import keras
from keras import layers
from keras import optimizers
import numpy as np
import glob
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import gridspec
import time

# GENERATOR MODEL
def make_generator(latent_dim):

    kernel_init = 'glorot_uniform'

    model = keras.Sequential()
    model.add(layers.Dense(512*4*4, input_dim=latent_dim))
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Reshape((4, 4, 512)))

    # Additional layer to improve performance of the generator model
    model.add(layers.Conv2DTranspose(filters=512, kernel_size=(4,4), strides=(1,1), padding='same', kernel_initializer=kernel_init))
    model.add(layers.BatchNormalization(momentum=0.9))
    model.add(layers.LeakyReLU(alpha=0.2))

    # from 4x4 to 8x8 resolution
    model.add(layers.Conv2DTranspose(filters=256, kernel_size=(4,4), strides=(2,2), padding='same', kernel_initializer=kernel_init))
    model.add(layers.BatchNormalization(momentum=0.9))
    model.add(layers.LeakyReLU(alpha=0.2))

    # from 8x8 to 16x16 resolution
    model.add(layers.Conv2DTranspose(filters=128, kernel_size=(4,4), strides=(2,2), padding='same', kernel_initializer=kernel_init))
    model.add(layers.BatchNormalization(momentum=0.9))
    model.add(layers.LeakyReLU(alpha=0.2))

    # from 16x16 to 32x32 resolution
    model.add(layers.Conv2DTranspose(filters=64, kernel_size=(4,4), strides=(2,2), padding='same', kernel_initializer=kernel_init))
    model.add(layers.BatchNormalization(momentum=0.9))
    model.add(layers.LeakyReLU(alpha=0.2))

    # Additional layer to improve performance of the generator model
    model.add(layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', kernel_initializer=kernel_init))
    model.add(layers.BatchNormalization(momentum=0.9))
    model.add(layers.LeakyReLU(alpha=0.2))

    # from 32x32 to 64x64 resolution
    model.add(layers.Conv2DTranspose(filters=3, kernel_size=(4,4), strides = (2,2), activation='tanh', padding='same', kernel_initializer=kernel_init))

    return model

# DISCRIMINATOR MODEL
def make_discriminator(in_shape=(64,64,3)):

    kernel_init = 'glorot_uniform'

    model = keras.Sequential()
    model.add(layers.Conv2D(filters=64, kernel_size=(4,4), strides=(2,2), padding='same', kernel_initializer=kernel_init, input_shape=in_shape))
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2D(filters=128, kernel_size=(4,4), strides=(2,2), padding='same', kernel_initializer=kernel_init))
    model.add(layers.BatchNormalization(momentum=0.9))
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2D(filters=256, kernel_size=(4,4), strides=(2,2), padding='same', kernel_initializer=kernel_init))
    model.add(layers.BatchNormalization(momentum=0.9))
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2D(filters=512, kernel_size=(4,4), strides=(1,1), padding='same', kernel_initializer=kernel_init))
    model.add(layers.BatchNormalization(momentum=0.9))
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))

    # Keeping the discriminator's learning rate higher than GAN's gave best results
    opt = optimizers.Adam(lr=0.0003, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

# GAN MODEL
def make_gan(g_model, disc_model):

    model = keras.Sequential()
    model.add(g_model)
    model.add(disc_model)

    opt = optimizers.Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model

def generate_latent_points(batch_size, latent_dim):
    return np.random.normal(0, 1, size=(batch_size, latent_dim))

def generate_fake_samples(g_model, batch_size, latent_dim):
    x_input = generate_latent_points(batch_size, latent_dim)
    X = g_model.predict(x_input)
    y = np.zeros((batch_size, 1))
    return X, y

def generate_real_samples(batch_size, image_shape, data_dir):
    X_dim = (batch_size, ) + image_shape
    X = np.empty(X_dim, dtype=np.float32)
    img_dir_list = list(glob.glob(data_dir))
    sample_imgs_paths = np.random.choice(img_dir_list, batch_size)
    for index, img_filename in enumerate(sample_imgs_paths):
        image = Image.open(img_filename)
        image = image.resize(image_shape[:-1])
        image = image.convert('RGB')
        # Augmenting the batch by horizontally flipping a randomly chosen image 
        if np.random.choice([True, False]):
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        image = np.asarray(image)
        # Normalizing input images
        image = (image/127.5)-1
        X[index,...] = image
    y = np.ones((batch_size, 1))
    y -= 0.2 * np.random.uniform(size=y.shape)
    return X, y

def train_gan(g_model, disc_model, gan_model, latent_dim, batch_size, image_shape, data_dir, n_epochs):

    disc_loss_list = list()
    gan_loss_list = list()

    for i in range(n_epochs): 
        epo_begin_time = time.time()
        for j in range(int(dataset_dim//batch_size)):

            X_real, y_real = generate_real_samples(batch_size, image_shape, data_dir)
            X_fake, y_fake = generate_fake_samples(g_model, batch_size, latent_dim)

            g_model.trainable=False

            d_loss1, _ = disc_model.train_on_batch(X_real, y_real)
            d_loss2, _ = disc_model.train_on_batch(X_fake, y_fake)

            g_model.trainable=True

            X_gan = generate_latent_points(batch_size, latent_dim)
            y_gan = np.ones((batch_size, 1))
            y_gan -= 0.2 * np.random.uniform(size=y_gan.shape)

            disc_model.trainable=False

            gan_loss = gan_model.train_on_batch(X_gan, y_gan)

            disc_model.trainable=True

            print('>%d, %d, Discriminator_real loss=%.3f, Discriminator_fake loss=%.3f GAN loss=%.3f' % (i+1, j+1, d_loss1, d_loss2, gan_loss))
 
        end_time = time.time()
        diff_time = int(end_time - epo_begin_time)
        print("Step %d completed. Time took: %s secs." % (i+1, diff_time))

        # Evaluating model's performance at the end of every epoch
        X_real, y_real = generate_real_samples(batch_size, image_shape, data_dir)
        X_fake, y_fake = generate_fake_samples(g_model, batch_size, latent_dim)

        d_loss1, _ = disc_model.evaluate(X_real, y_real, verbose=0)
        d_loss2, _ = disc_model.evaluate(X_fake, y_fake, verbose=0)

        X_gan = generate_latent_points(batch_size, latent_dim)
        y_gan = np.ones((batch_size, 1))
        y_gan -= 0.2 * np.random.uniform(size=y_gan.shape)

        gan_loss = gan_model.evaluate(X_gan, y_gan, verbose=0)

        disc_loss_list.append(0.5*(d_loss1 + d_loss2))
        gan_loss_list.append(gan_loss)

        generate_images(g_model, batch_size, latent_dim, img_save_dir+"/"+"control_image_"+str(i+1)+".png")

        # save weights every 10 epochs
        if (i+1)%10 == 0:
            disc_model.save_weights('saved_models/discriminator_weights_%d' % (i+1))
            g_model.save_weights('saved_models/generator_weights_%d' % (i+1))
 
    evaluation_plot(disc_loss_list, gan_loss_list)
    g_model.save('saved_models/generator.h5')

def generate_images(generator, batch_size, latent_dim, save_dir):
    noise = generate_latent_points(batch_size, latent_dim)
    fake_data_X = generator.predict(noise)
    plt.figure(figsize=(4,4))
    rand_indices = np.random.choice(fake_data_X.shape[0], 16, replace=False)
    gspec = gridspec.GridSpec(4, 4, wspace=0, hspace=0)
    for i in range(16):
        ax = plt.subplot(gspec[i])
        ax.set_aspect('equal')
        image = fake_data_X[rand_indices[i],:,:,:]
        plt.imshow(((image+1)*127.5).astype(np.uint8))
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_dir, bbox_inches='tight', pad_inches=0)

def evaluation_plot(disc_loss_list, gan_loss_list):
    fig = plt.figure()
    plt.plot(disc_loss_list, label="Discriminator model")
    plt.plot(gan_loss_list, label = "GAN model")
    plt.legend()
    plt.title("Losses")
    plt.savefig("evaluation_plot.png", bbox_inches='tight', pad_inches=0)

noise_dim = 100
image_shape = (64, 64, 3)
n_epochs = 50
batch_size = 32
data_dir = r"cropped_images\*.jpg"
img_save_dir = "generated_img"
dataset_dim = len(list(glob.glob(data_dir)))

discriminator = make_discriminator()
generator = make_generator(noise_dim)
gan_model = make_gan(generator, discriminator)

train_gan(generator, discriminator, gan_model, noise_dim, batch_size, image_shape, data_dir, n_epochs)