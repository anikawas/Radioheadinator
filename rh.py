import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, losses
import os


mel_spectrograms_dir = 'mel_spectrograms'  # Change to your directory

# Determine the maximum length of mel-spectrograms
max_length = 0
for file_name in os.listdir(mel_spectrograms_dir):
    if file_name.endswith(".npy"):
        file_path = os.path.join(mel_spectrograms_dir, file_name)
        mel_spectrogram = np.load(file_path)
        if mel_spectrogram.shape[1] > max_length:
            max_length = mel_spectrogram.shape[1]

fixed_length = max_length
print(f"Fixed length determined: {fixed_length}")


def load_and_pad_mel_spectrograms(mel_spectrograms_dir, fixed_length):
    mel_spectrograms = []
    for file_name in os.listdir(mel_spectrograms_dir):
        if file_name.endswith(".npy"):
            file_path = os.path.join(mel_spectrograms_dir, file_name)
            mel_spectrogram = np.load(file_path)
            if mel_spectrogram.shape[1] < fixed_length:
                # Pad the mel-spectrogram to the fixed length
                padding = fixed_length - mel_spectrogram.shape[1]
                mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, padding)), mode='constant')
            else:
                # Truncate the mel-spectrogram to the fixed length
                mel_spectrogram = mel_spectrogram[:, :fixed_length]
            mel_spectrograms.append(mel_spectrogram)
    return np.array(mel_spectrograms)

# Load and pad mel-spectrograms to the fixed length
mel_spectrograms_array = load_and_pad_mel_spectrograms(mel_spectrograms_dir, fixed_length)
mel_spectrograms_array = np.expand_dims(mel_spectrograms_array, axis=-1)  # Add channel dimension


def build_encoder(input_shape, latent_dim):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    z_mean = layers.Dense(latent_dim)(x)
    z_log_var = layers.Dense(latent_dim)(x)
    return models.Model(inputs, [z_mean, z_log_var])

def build_decoder(latent_dim, output_shape):
    latent_inputs = layers.Input(shape=(latent_dim,))
    
    intermediate_dim = (output_shape[0]//4, output_shape[1]//4, output_shape[2])
    num_intermediate_elements = np.prod(intermediate_dim)
    
    x = layers.Dense(num_intermediate_elements, activation='relu')(latent_inputs)
    x = layers.Reshape(intermediate_dim)(x)
    x = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    outputs = layers.Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')(x)
    return models.Model(latent_inputs, outputs)

class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        z_mean, z_log_var = self.encoder(inputs)
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
        return self.decoder(z)

input_shape = (128, fixed_length, 1)  # Updated input shape
latent_dim = 16
encoder = build_encoder(input_shape, latent_dim)
decoder = build_decoder(latent_dim, input_shape)
vae = VAE(encoder, decoder)


def vae_loss(inputs, outputs):
    reconstruction_loss = tf.keras.losses.MeanSquaredError()(inputs, outputs)
    reconstruction_loss *= input_shape[0] * input_shape[1]
    z_mean, z_log_var = vae.encoder(inputs)
    kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
    kl_loss = tf.reduce_mean(kl_loss) * -0.5
    return reconstruction_loss + kl_loss

vae.compile(optimizer='adam', loss=vae_loss)


mel_spectrograms_array = mel_spectrograms_array[:100] 
vae.fit(mel_spectrograms_array, mel_spectrograms_array, epochs=50, batch_size=4)
