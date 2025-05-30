import tensorflow as tf
import tensorflow.keras.layers as tkl
import numpy as np



from .vae_SOLUTIONS import sample_from_normal, kl_divergence, Encoder, Decoder
from .gan_SOLUTIONS import Discriminator

# VAE~GAN with 3 conv layers/module (enc & dec), pixel recon loss, non conditional
class VAEGAN(tf.keras.Model):
    
    # These are CLASS attributes, which we create here in the class namespace,
    # outside of any method. Unlike INSTANCE attributes, which are created
    # inside __init__ or other methods, the values of these attributes are
    # constant across all instances. Class attributes are often used to define
    # constants or default values. Here, we're using class attributes to tell
    # VAEGAN which classes correspond to the 3 submodels. This makes it easier
    # to switch them out later if we modify the architectures and come up with
    # different submodel classes (see VAEGANLarger below).
    encoder_class = Encoder
    decoder_class = Decoder
    discriminator_class = Discriminator
    
    def __init__(self, 
                 n_latent_dims=8, 
                 image_shape=(32, 32, 1),
                 recon_loss_weight=1.,
                 kl_loss_weight=1.,
                 adv_loss_weight=1.,
                 encoder_params = {},
                 decoder_params = {},
                 discriminator_params = {},
                 name='vaegan', 
                 **kwargs):
        """Variational autoencoder-generative adversarial network. Contains a
        VAE which learns to compress and decompress an image and a
        discriminator which helps the VAE produce more realistic images.

        Args:
            n_latent_dims (int, optional): Size of latent representation.
                Defaults to 8. 
            image_shape (tuple, optional): Image shape. Defaults
                to (32, 32, 1). 
            recon_loss_weight (float, optional): Weight for reconstruction 
                loss. Defaults to 1.. 
            kl_loss_weight (float, optional): Weight for KL divergence 
                regularization in the encoder. Defaults to 1.. 
            adv_loss_weight (float, optional): Weight for adversarial loss in 
                the decoder. Defaults to 1.. 
            name (str, optional): Model name. Defaults to 'vaegan'.
        """        
        
        super(VAEGAN, self).__init__(name=name, **kwargs)
        
        self.n_latent_dims = n_latent_dims
        self.image_shape = image_shape
        self.recon_loss_weight = recon_loss_weight
        self.kl_loss_weight = kl_loss_weight
        self.adv_loss_weight = adv_loss_weight
        
        self.encoder = self.encoder_class(n_latent_dims=self.n_latent_dims, **encoder_params)
        self.decoder = self.decoder_class(image_shape=self.image_shape, **decoder_params)
        self.discriminator = self.discriminator_class(**discriminator_params)
        
        # Use the mean squared error as the reconstruction loss. Do not reduce
        # values (average over samples), return per-pixel values instead.       
        self.loss_recon = tf.keras.losses.MeanSquaredError(name='recon_mse', reduction='none')
        
        # Use binary cross-entropy for the discrminator's classification loss
        self.loss_disc = tf.keras.losses.BinaryCrossentropy(name='disc_bce')
        
        # Define some custom metrics to track the running means of each loss.
        # The values of these metrics will be printed in the progress bar with
        # each training iteration.
        self.loss_recon_tracker = tf.keras.metrics.Mean(name='recon_loss')
        self.loss_kl_tracker = tf.keras.metrics.Mean(name='kl_loss')
        self.loss_enc_tracker = tf.keras.metrics.Mean(name='enc_loss')
        self.loss_dec_tracker = tf.keras.metrics.Mean(name='dec_loss')
        self.loss_disc_tracker = tf.keras.metrics.Mean(name='disc_loss')

    def call(self, inputs, training=None):
        z_mean, z_logvar = self.encoder(inputs, training=training)
        z = sample_from_normal(z_mean, z_logvar)
        recons = self.decoder(z, training=training)
        return recons
                
    @property
    def metrics(self):
        '''
        Return a list of losses
        '''
        return [self.loss_recon_tracker,
                self.loss_kl_tracker,
                self.loss_enc_tracker,
                self.loss_dec_tracker,
                self.loss_disc_tracker]
    
    def train_step(self, data):
        """Defines a single training iteration, including the forward pass,
        computation of losses, backpropagation, and weight updates.

        Args:
            data (tensor): Input images.

        Returns:
            dict: Loss values.
        """        
        images_real = data
        
        # persistent=True is required to compute multiple gradients from a single GradientTape
        with tf.GradientTape(persistent=True) as gt:
            # Use encoder to predict probabilistic latent representations 
            z_mean, z_logvar = self.encoder(images_real, training=True)
            
            # Sample a point from the latent distributions. 
            z = sample_from_normal(z_mean, z_logvar)
            
            # Use decoder to reconstruct image
            recons = self.decoder(z, training=True)
            
            # Compute KL divergence loss between latent representations and the prior
            kl_loss = kl_divergence(z_mean, z_logvar)
            
            # Compute reconstruction loss
            recon_loss_pixel = self.loss_recon(images_real, recons)
            # Recon loss is computed per pixel. Sum over the pixels and then
            # average across samples.
            recon_loss_sample = tf.reduce_sum(recon_loss_pixel, axis=(1, 2))
            recon_loss = tf.reduce_mean(recon_loss_sample)
            
            # Synthesize some new images by having the decoder generate from
            # random latent vectors.
            n_samples = tf.shape(images_real)[0]
            z_random = tf.random.normal((n_samples, self.n_latent_dims))
            images_fake = self.decoder(z_random, training=True)
            
            # Create label vectors, 1 for real and 0 for fake/reconstruction images
            labels_real = tf.ones((n_samples, 1))
            labels_fake = tf.zeros((n_samples, 1))
            
            # Concatenate real, reconstruction, and fake images 
            images_concat = tf.concat([images_real, recons, images_fake], axis=0)
            labels_concat = tf.concat([labels_real, labels_fake, labels_fake], axis=0)
            
            # Predict with the discriminator
            labels_pred = self.discriminator(images_concat, training=True)
            
            # Compute discriminator classification loss
            disc_loss = self.loss_disc(labels_concat, labels_pred)
            
            # Encoder loss includes the KL divergence and the reconstruction loss
            encoder_loss = kl_loss * self.kl_loss_weight + recon_loss * self.recon_loss_weight
            
            # Decoder loss includes the reconstruction loss and adversarial loss
            # (negative discriminator loss since the decoder wants the
            # discriminator to predict wrongly)
            decoder_loss = recon_loss * self.recon_loss_weight - disc_loss * self.adv_loss_weight
                       
        # Compute the gradients for each loss wrt their respectively model weights
        grads_enc = gt.gradient(encoder_loss, self.encoder.trainable_weights)
        grads_dec = gt.gradient(decoder_loss, self.decoder.trainable_weights)
        grads_disc = gt.gradient(disc_loss, self.discriminator.trainable_weights)
        
        # Apply the gradient descent steps to each submodel. The optimizer
        # attribute is created when model.compile(optimizer) is called by the
        # user.
        self.optimizer.apply_gradients(zip(grads_enc, self.encoder.trainable_weights))
        self.optimizer.apply_gradients(zip(grads_dec, self.decoder.trainable_weights))
        self.optimizer.apply_gradients(zip(grads_disc, self.discriminator.trainable_weights))           
        
        # Update the running means of the losses
        self.loss_recon_tracker.update_state(recon_loss)
        self.loss_kl_tracker.update_state(kl_loss)
        self.loss_enc_tracker.update_state(encoder_loss)
        self.loss_dec_tracker.update_state(decoder_loss)
        self.loss_disc_tracker.update_state(disc_loss)
        
        # Get the current values of these running means as a dict. These values
        # will be printed in the progress bar.
        dictLosses = {loss.name: loss.result() for loss in self.metrics}
        return dictLosses

    def get_config(self):
        # To allow saving and loading of a custom model, we need to implement a
        # get_config method. This should return a dict containing all important
        # arguments needed to instantiate the model.
        return {'n_latent_dims': self.n_latent_dims,
                'image_shape': self.image_shape,
                'recon_loss_weight': self.recon_loss_weight,
                'kl_loss_weight': self.kl_loss_weight,
                'adv_loss_weight': self.adv_loss_weight}
    
    @classmethod
    def from_config(cls, config):
        # Create model using config, which is a dict of arguments. Note that cls
        # refers to the current class, and calling cls is equivalent to
        # instantiating the class.
        return cls(**config)
    

    