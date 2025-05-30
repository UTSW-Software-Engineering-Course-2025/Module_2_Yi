import tensorflow as tf
import tensorflow.keras.layers as tkl
import numpy as np

def fParseActivationFunctions(lsActFuncs:list):
    """ convert list of activation functions into (string, tf.nn.xxxx) pairs for layer naming
    Note, original index of activations are preserved

    Non-aliased activation functions supported:
    - leaky_relu
    - PReLU

    Args:
        lsActFuncs (list): list of activation functions to parse
    
    Returns: 
        dictActivationFunctions (dict): dictionary of parsed activation functions
            format: {0 : (`ActivationFunctionName`, `activationFunction`), ...}
    """
    idx = 0
    dictActivationFunctions = {}
    for f in lsActFuncs:
        # If the activation function is passed in as a string, assume tf has alias
        # i.e. lsActFunc = ['relu']
        if isinstance(f, str):
            dictActivationFunctions[idx] = (f, f)
        # Check for non-string aliased activation functions
        elif f == tf.nn.leaky_relu:
            dictActivationFunctions[idx] = ('leaky_relu', f)
        #elif f == tkl.PReLU:
        #    dictActivationFunctions[idx] = ('PReLU', f)
        # Error handling
        else:
            raise ValueError(f'You entered {f} which is not currently accepted as an activation function') 
        idx += 1
    return dictActivationFunctions





class Generator(tf.keras.Model):
    
    def __init__(self,
                 generator_conv_layer_units = [128,128],
                 generator_conv_layer_kernel_size = [4,4],
                 generator_conv_layer_strides = [(2,2), (2,2)],
                 generator_conv_layer_activation_functions = [tf.nn.leaky_relu,tf.nn.leaky_relu],
                 discriminator_dense_layer_units = [128],
                 generator_dense_layer_units = [],
                 image_shape=(32, 32, 1),
                 name='generator',
                 **kwargs):
        """Convolutional generator which produces an image from a latent vector.

        Args:
            image_shape (tuple, optional): Image shape. Defaults to (32, 32, 1).
            name (str, optional): Model name. Defaults to 'generator'.
        """        
        # Call to tf.keras.Model's __init__ method. Does some model setup like
        # setting the name.
        super().__init__(name=name, **kwargs)
        
        self.image_shape = image_shape

        '''
        First, we need to figure out how many neurons the initial Dense layer
        needs in order to produce the desired output image shape.
        
        The first few layers of the generator look like this: latent vector ->
        dense -> reshape 1D vector to 3D tensor -> conv0
        
        There are 2 convolutions with 2 x 2 striding (plus 1 non-strided
        convolution). That means that the input to layer conv0 should have shape 
        
        (d/2^2) x (d/2^2) x (n_filters)
        
        Consequently, the first Dense layer needs to output a vector with this
        number of elements. 
            
        '''
        
        n_first_layer_filters = generator_conv_layer_units[-1] # Number of filters/channels in the input to first convolutional layer
        n_strided_layers = sum([strides != (1,1) for strides in generator_conv_layer_strides])
        first_conv_input_shape = (self.image_shape[0] // (2 ** n_strided_layers),
                                  self.image_shape[1] // (2 ** n_strided_layers),
                                  n_first_layer_filters)
        
        dense_neurons = np.prod(first_conv_input_shape)
        
        # Parse activation function lists for 
        self.generator_conv_layer_activation_functions = fParseActivationFunctions(generator_conv_layer_activation_functions)
        # Init empty container to hold layers
        self.generator_layers = []

        # Dense Layer blocks
        # Fill layers in reverse so same list can be used for discriminator
        for i in range(len(generator_dense_layer_units)-1, -1, -1):
            self.generator_layers += [tkl.Dense(generator_dense_layer_units[i],name = f"dense{i}")]
            self.generator_layers += [tkl.ReLU(name=f'relu{i}')]

        # Reshape
        self.generator_layers += [tkl.Dense(dense_neurons, name='dense_to_conv')]
        self.generator_layers += [tkl.Activation(tf.nn.leaky_relu, name=f'leaky_relu_dense_to_conv')]
        self.generator_layers += [tkl.Reshape(first_conv_input_shape, name='reshape')]
        
        # Conv layer blocks
        # Fill layers in reverse so same list can be used for encoder
        for i in range(len(generator_conv_layer_units)-1, -1, -1):
            self.generator_layers += [tkl.Conv2DTranspose(generator_conv_layer_units[i], generator_conv_layer_kernel_size[i], padding = 'same', 
                                       strides = generator_conv_layer_strides[i], name = f"tconv{i}")]
            self.generator_layers += [tkl.Activation(self.generator_conv_layer_activation_functions[i][1],name=f'{self.generator_conv_layer_activation_functions[i][0]}{i}')]
                                    
        # A final convolution layer with a sigmoid activation to produce the
        # output image. The number of filters should equal the number of color
        # channels in the image.
        n_channels = self.image_shape[-1]
        self.conv_out = tkl.Conv2D(n_channels, 3, padding='same', name='conv_out')
        self.sigmoid_out = tkl.Activation('sigmoid', name='sigmoid_out')
        
    def call(self, inputs, training=None):
        """Forward pass.

        Args:
            inputs (tensor): Batch of inputs containing a compressed representation.
            training (bool, optional): Whether the model is training or 
                testing model. Defaults to None.
                
        Returns:
            tensor: reconstructed image
        """                             

        x = inputs
        for layer in self.generator_layers:
            x = layer(x, training = training)
  
        x = self.conv_out(x)
        x = self.sigmoid_out(x)
        
        return x    
    
    def get_config(self):
        # To allow saving and loading of a custom model, we need to implement a
        # get_config method. This should return a dict containing all important
        # arguments needed to instantiate the model.
        return {'image_shape': self.image_shape}
    
    @classmethod
    def from_config(cls, config):
        # Create model using config, which is a dict of arguments. Note that cls
        # refers to the current class, and calling cls is equivalent to
        # instantiating the class.
        return cls(**config)
    

class Discriminator(tf.keras.Model):
    def __init__(self, 
                 discriminator_conv_layer_units = [32,64],
                 discriminator_conv_layer_kernel_size = [4,4],
                 discriminator_conv_layer_strides = [(1,1), (2,2)],
                 discriminator_conv_layer_activation_functions = ['relu','relu'],
                 discriminator_dense_layer_units = [],
                 generator_dense_layer_units = [],
                 name='discriminator', 
                 **kwargs):
        """Discriminator for classifying real from fake images.

        Args:
            name (str, optional): Model name. Defaults to 'discriminator'.
        """        
        super(Discriminator, self).__init__(name=name, **kwargs)
        
        # Parse activation function lists for 
        self.discriminator_conv_layer_activation_functions = fParseActivationFunctions(discriminator_conv_layer_activation_functions)
        self.discriminator_layers = []

         # Conv layer blocks
        for i in range( len(discriminator_conv_layer_units) ):
            self.discriminator_layers += [tkl.Conv2D(discriminator_conv_layer_units[i], discriminator_conv_layer_kernel_size[i], padding = 'same', 
                                       strides = discriminator_conv_layer_strides[i], name = f"conv{i}")]
            self.discriminator_layers += [tkl.BatchNormalization(name=f'bn{i}')]
            self.discriminator_layers += [tkl.Activation(self.discriminator_conv_layer_activation_functions[i][1],
                                                         name=f'{self.discriminator_conv_layer_activation_functions[i][0]}{i}')]
            
        # Flatten 2D output into a vector, then apply dense layer(s)
        self.discriminator_layers += [tkl.Flatten(name='flatten')]
        
        # Dense Layer blocks
        for i in range(len(discriminator_dense_layer_units)):
            self.discriminator_layers += [tkl.Dense(discriminator_dense_layer_units[i],name = f"dense{i}")]

        # Classification output layer
        self.dense_out = tkl.Dense(1, name='dense_out')
        self.sigmoid_out = tkl.Activation('sigmoid', name='sigmoid_out')
        #self.discriminator_layers += [tkl.Dense(1, name = f"dense_out", activation = 'sigmoid')]

        
    def call(self, inputs, training=None):
        """Define the forward pass, which produces a predicted classification.

        Args:
            inputs (tensor): Image input.
            training (bool, optional): Whether the model is training or 
                testing model. Defaults to None.
                
        Returns:
            tensor: predicted classification
        """        
        x = inputs
        for layer in self.discriminator_layers:
            x = layer(x, training = training)
        y = self.dense_out(x)
        y = self.sigmoid_out(y)
        return y
        
            
    def get_config(self):
        # To allow saving and loading of a custom model, we need to implement a
        # get_config method. This should return a dict containing all important
        # arguments needed to instantiate the model.
        return {}
    
    @classmethod
    def from_config(cls, config):
        # Create model using config, which is a dict of arguments. Note that cls
        # refers to the current class, and calling cls is equivalent to
        # instantiating the class.
        return cls(**config)


class GAN(tf.keras.Model):
    '''Generative adversarial network'''


    def __init__(self, 
                 n_latent_dims=128, 
                 image_shape=(32, 32, 1),
                 generator_params = {},
                 discriminator_params = {},
                 generator_lr=0.0001,
                 discriminator_lr=0.00001,
                 name='gan', 
                 **kwargs):
        """Generative adversarial network containing a generator to synthesize 
        images and an adversary to to discriminate between real and fake images.

        Args:
            n_latent_dims (int, optional): Size of latent representation.
                Defaults to 8. 
            image_shape (tuple, optional): Image shape. Defaults
                to (32, 32, 1). 
            generator_lr (float, optional): Adam learning rate for generator. 
                Defaults to 0.0001.
            discriminator_lr (float, optional): Adam learning rate for discriminator. 
                Defaults to 0.00001.
            name (str, optional): Model name. Defaults to 'gan'.
        """        
        
        # ===== ToImplement   Exercise5c ====
        # ===================================  
        # 
        # 1. Call base class constructor      
        super().__init__(name=name, **kwargs)
        
        # 2. store the 4 necessary attributes provided to the constructor. 
        #    Use as your attribute names, the same names as in the constructor input arguments
        self.n_latent_dims = n_latent_dims
        self.image_shape = image_shape
        self.generator_lr = generator_lr
        self.discriminator_lr = discriminator_lr
        
        # 3. Use the has-a mechanism to contain two construct necessary instances and store them in self.generator and self.discriminator       
        self.generator = Generator(**generator_params)
        self.discriminator = Discriminator(**discriminator_params)
                
        # 4. Use binary cross-entropy for the discrminator's classification loss. Store this in self.loss_bce
        # Look up the keras function losses.BinaryCrossentropy. Give it a suitable name.
        self.loss_bce = tf.keras.losses.BinaryCrossentropy(name='bce')
        
        # 5. Create a custom metric objects to track the running means of each loss.
        # The values will be printed in the progress bar with each training
        # iteration.
        # store them in self.loss_gen_tracker and self.loss_disc_tracker.
        # Find a suitable metric from tf.keras.metrics
        self.loss_gen_tracker = tf.keras.metrics.Mean(name='gen_loss')
        self.loss_disc_tracker = tf.keras.metrics.Mean(name='disc_loss')
        
        # 6. Create Adam optimizers to do the gradient descent 
        #    Store them in self.optimizer_gen and self.optimizer_disc
        self.optimizer_gen = tf.keras.optimizers.Adam(lr=generator_lr)
        self.optimizer_disc = tf.keras.optimizers.Adam(lr=discriminator_lr)

        print(f"Loaded version: {__name__}")

    # ===== ToImplement   Exercise5d ====
    # ===================================
    # Implement the GAN forward pass method
    #  0. Write an appropriate doc string
    #  1. there should be 3 arguments to this method. 
    #     one of them is inputs  which contains the data to do the forward pass upon.
    #  2. Set n =  the number of samples by selecting the appropriate dimension using tf.shape
    #  3. Sample from a normal distribution, see tf.random documentation for a suitable function
    #     you will want n x n_latent_dims  samples from this normal distn.
    #  4. To generate fake images pass that sample through the forward pass of the generator, but also tell it whether we are in training mode or not, (from the GAN forward pass method input argument).
    #  5. Return the generated fake images.             
    def call(self, inputs, training=None):


        n_samples = tf.shape(inputs)[0]
        
        z_random = tf.random.normal((n_samples, self.n_latent_dims))
        images_fake = self.generator(z_random, training=training)
        
        return images_fake
        
    @property
    def metrics(self):
        '''
        Return a list of losses
        '''
        return [self.loss_gen_tracker,
                self.loss_disc_tracker]
    
    # ===== ToImplement   Exercise5e ====
    # ===================================
    # Implement the training step method, since GANs require specialized training.
    # 0. Write an appropriate doc string 
    # 1. there are two input arguments. One of them is the tensor, images_real, with the current minibatch to train upon.
    def train_step(self, data):
        """Defines a single training iteration, including the forward pass,
        computation of losses, backpropagation, and weight updates.

        Args:
            data (tensor): Input images.

        Returns:
            dict: Loss values.
        """        

      
        images_real = data
        
        # Part 1: Train the discriminator
        # 2. Generate images from random latent vectors.
        #  2a) Set n =  the number of samples by selecting the appropriate dimension using tf.shape
        #  2b) Sample from a normal distribution, see tf.random documentation for a suitable function
        #     you will want n x n_latent_dims  samples from this normal distn.  
        #  2c) Use the generator forward pass, i.e, via self.generator() to create a variable with fake images
        #     (images_fake) using the sample from the normal distn you just created in the previous substep, 2b). 
        #     Also specify that training=False 
        # 
        n_samples = tf.shape(images_real)[0]
        z_random = tf.random.normal((n_samples, self.n_latent_dims))
        images_fake = self.generator(z_random, training=False)
        
        # 3. Create label vectors varaible, labels_real, containing ones
        #   Also create a label vector, labels_fake, containing zeros.
        #    e.g. hint for part you may want to use tf.ones
        labels_real = tf.ones((n_samples, 1))
        labels_fake = tf.zeros((n_samples, 1))

        # 4. Concatenate real and fake images into new variable,  images_disc. Hint: See tf.concat
        #    Also concatenate real and fake labels into new varaible, labels_disc.
        images_disc = tf.concat([images_real, images_fake], axis=0)
        labels_disc = tf.concat([labels_real, labels_fake], axis=0)
        
        # 5. start a GraidentTape with default arguments
        with tf.GradientTape() as gt:
            # 6. indented: Predict with the discriminator's forward pass via self.discriminator()
            #    store the predictions in variable, labels_pred 
            #    Specify that now, training=True
            labels_pred = self.discriminator(images_disc, training=True)
            
            # 7. indented: Compute discriminator classification loss
            #    in a new variable, disc_loss, compute the binary cross entropy loss using one of the attributes of self
            disc_loss = self.loss_bce(labels_disc, labels_pred)
                                   
        # 8. NOT indented: Compute the gradient of the lost wrt the discriminator weights
        #  in a new variable, grads_disc store the gradients of disc_loss with respect to the discriminator's trainable weights  
        grads_disc = gt.gradient(disc_loss, self.discriminator.trainable_weights)
        
        # 9. Apply the weight updates
        #    use self.optimizer_disc to apply the weight updates. Hint: use zip
        self.optimizer_disc.apply_gradients(zip(grads_disc, self.discriminator.trainable_weights))           
        
        # Part 2: Train the generator
        # 10. start a GraidentTape with default arguments. Use a different tape context variable than used above.           
        with tf.GradientTape() as gt2:
            # [ 11. indented: ] Generate images from random latent vectors. Generate twice as many
            # images as the batch size so that the generator sees as many
            # samples as the discriminator did. 
            #  To help you along, we give you this part 
            z_random = tf.random.normal((n_samples * 2, self.n_latent_dims))
            images_fake = self.generator(z_random, training=True)
                        
            # 12. indented: Predict with the discriminator's forward pass the labels for images_fake
            #     set training=False. Store the prediction results in new variable, labels_pred
            labels_pred = self.discriminator(images_fake, training=False)
            
            # 13. indented: We want to the discriminator to think these images are real, so we
            # calculate the loss between these predictions and the "real image" labels
            # 13a)  build a new variable, labels_gen, which is a vector of ones of size 2*n_samples x 1
            # 13b)  build a new variable, gen_loss containing the binary cross entropy loss.
            labels_gen = tf.ones((2 * n_samples, 1))
            gen_loss = self.loss_bce(labels_gen, labels_pred)
            
        # 14. NOT indented: compute the gradient of the lost wrt the generator weights
        #  in a new variable, grads_gen store the gradients of gen_loss with respect to the generator's trainable weights         
        grads_gen = gt2.gradient(gen_loss, self.generator.trainable_weights)
        
        # 15. Apply the weight updates
        #    use self.optimizer_gen to apply the weight updates. 
        self.optimizer_gen.apply_gradients(zip(grads_gen, self.generator.trainable_weights))                
        
        # 16. Update the running means of the losses including loss_gen_tracker and loss_disc_tracker
        self.loss_gen_tracker.update_state(gen_loss)
        self.loss_disc_tracker.update_state(disc_loss)
        
        # [ 17. ] Get the current values of these running means as a dict. These values
        # will be printed in the progress bar.
        # To help you along this is given.
        dictLosses = {loss.name: loss.result() for loss in self.metrics}

        # return the dictionary of losses
        return dictLosses

    def get_config(self):
        # To allow saving and loading of a custom model, we need to implement a
        # get_config method. This should return a dict containing all important
        # arguments needed to instantiate the model.
        return {'n_latent_dims': self.n_latent_dims,
                'image_shape': self.image_shape,
                'generator_lr': self.generator_lr,
                'discriminator_lr': self.discriminator_lr}
    
    @classmethod
    def from_config(cls, config):
        # Create model using config, which is a dict of arguments. Note that cls
        # refers to the current class, and calling cls is equivalent to
        # instantiating the class.
        return cls(**config)
    
class MultiTaskDiscriminator(Discriminator):
    
    def __init__(self, 
                 n_classes,
                 discriminator_conv_layer_units = [32,64],
                 discriminator_conv_layer_kernel_size = [4,4],
                 discriminator_conv_layer_strides = [(1,1), (2,2)],
                 discriminator_conv_layer_activation_functions = ['relu','relu'],
                 discriminator_dense_layer_units = [],
                 generator_dense_layer_units = [],
                 name='discriminator', 
                 **kwargs):
        """Discriminator for classifying real from fake/reconstructed images.

        Args:
            name (str, optional): Model name. Defaults to 'discriminator'.
        """        
        super(Discriminator, self).__init__(name=name, **kwargs)
        
        self.n_classes = n_classes
        
        # Define the layers for the Discriminator module that we will later link together in the call() method.
                # Parse activation function lists for 
        self.discriminator_conv_layer_activation_functions = fParseActivationFunctions(discriminator_conv_layer_activation_functions)
        self.discriminator_layers = []

         # Conv layer blocks
        for i in range( len(discriminator_conv_layer_units) ):
            self.discriminator_layers += [tkl.Conv2D(discriminator_conv_layer_units[i], discriminator_conv_layer_kernel_size[i], padding = 'same', 
                                       strides = discriminator_conv_layer_strides[i], name = f"conv{i}")]
            self.discriminator_layers += [tkl.BatchNormalization(name=f'bn{i}')]
            self.discriminator_layers += [tkl.Activation(self.discriminator_conv_layer_activation_functions[i][1],
                                                         name=f'{self.discriminator_conv_layer_activation_functions[i][0]}{i}')]
            
        # Flatten 2D output into a vector, then apply dense layer(s)
        self.discriminator_layers += [tkl.Flatten(name='flatten')]
        
        # Dense Layer blocks
        for i in range(len(discriminator_dense_layer_units)):
            self.discriminator_layers += [tkl.Dense(discriminator_dense_layer_units[i],name = f"dense{i}")]

        # Layers needed for block 4a: this predicts whether the image is real or fake
        # 1. self.dense_real ... define a dense layer via the Dense() function with 1 neauron and the name dense_real
        # 2. self.sigmoid_real ... define a sigmoid activation function layer via the Activation() function with the 'sigmoid' method and name sigmoid_real
        self.dense_real = tkl.Dense(1, name='dense_real')
        self.sigmoid_real = tkl.Activation('sigmoid', name='sigmoid_real')
        
        # Layers needed for block 4b: this predicts the class label of the image 
        # 1. self.dense_class ... define a dense layer via the Dense() function with n_classes and the name dense_class
        # 2. self.softmax_class  ... define a softmax activation  layer via the Softmax() function with the name softmax_class
        self.dense_class = tkl.Dense(self.n_classes, name='dense_class')
        self.softmax_class = tkl.Softmax(name='softmax_class')


    def call(self, inputs, training=None, return_features=False):
        """Define the forward pass, which produces a predicted classification.

        Args:
            inputs (tensor): Image input.
            training (bool, optional): Whether the model is training or 
                testing model. Defaults to None.
            return_features (bool, optional): Return the output of the last 
                convolutional layer. Defaults to False.
                
        Returns:
            tensor: predicted classification
        """        
        x = inputs
        for layer in self.discriminator_layers:
            x = layer(x, training = training)
        
        # Block 4a  x --> dense_real --> y_real;    y_real --> sigmoid_real --> y_real 
        # 1. use the dense_real layer to apply a dense NN layer that learns the combinations of elements in x that are predictive of authenticity. Store the result in y_real.
        # 2. Use the sigmoid_real layer to map the information stored in y_real to a probabilistic autheticity prediction of real (1.0) or fake (0.0)
        y_real = self.dense_real(x)
        y_real = self.sigmoid_real(y_real)
        
        # Block 4a  x --> dense_real --> y_real;    y_real --> sigmoid_real --> y_real 
        # 1. use the dense_class layer to apply a dense NN layer to learn the combinations of elements in x that are predictive of class label. Store the result in y_class.
        # 2. Use the softmax_class layer to map the information stored in y_class to a prediction of the class label (one-hot encoded). Store the result in y_class
        y_class = self.dense_class(x)
        y_class = self.softmax_class(y_class)
        
        if return_features:
            return y_real, y_class, x
        else:
            return y_real, y_class
    
    def get_config(self):
        return {'n_classes': self.n_classes}
    

class ConditionalGAN(GAN):
    '''Conditional generative adversarial network'''
    
    def __init__(self, 
                 n_classes=10,
                 cond_loss_weight=1.,
                 generator_params = {},
                 discriminator_params = {},
                 name='cgan', 
                 **kwargs):
        """Generative adversarial network containing a generator to synthesize 
        images and an adversary to to discriminate between real and fake images.

        Args:
            n_classes (int, optional): Number of classes. Defaults to 10. 
            cond_loss_weight (float, optional): Weight of conditional loss for 
                generator. Defaults to 1.
            name (str, optional): Model name. Defaults to 'cgan'.
        """          
        
        # ToImplement Exercise6a_part1 ==
        # ===============================
        super().__init__(name=name, generator_params=generator_params, **kwargs)

        self.n_classes = n_classes
        self.cond_loss_weight = cond_loss_weight
        
        self.discriminator = MultiTaskDiscriminator(n_classes, **discriminator_params)
        
        # Categorical cross-entropy loss for discriminator classification
        self.loss_class = tf.keras.losses.CategoricalCrossentropy(name='class_cce')
        
        # Classification accuracy 
        self.metric_class = tf.keras.metrics.TopKCategoricalAccuracy(k=1, name='top_1_acc')
     
        print(f"Loaded version: {__name__}")

    def call(self, inputs, training=None):
        # ToImplement Exercise6a_part2 ==
        # ===============================

        _, classes = inputs
        n_samples = tf.shape(classes)[0]
        z_random = tf.random.normal((n_samples, self.n_latent_dims))
        z_random_conditional = tf.concat([z_random, classes], axis=1)
        images_fake = self.generator(z_random_conditional, training=training)
        
        return images_fake
        
    @property
    def metrics(self):
        '''
        Return a list of losses
        '''
        return [self.loss_gen_tracker,
                self.loss_disc_tracker,
                self.metric_class]
    
    def generate_random_classes(self, n_samples):
        # Generate random class labels from uniform distribution
        class_probs = tf.ones((1, self.n_classes))
        class_random = tf.random.categorical(class_probs, num_samples=n_samples)
        class_random = tf.squeeze(class_random)
        # Convert from categorical to one-hot
        class_random = tf.one_hot(class_random, depth=self.n_classes, dtype=tf.float32)
        return class_random
    
    def train_step(self, data):
        """Defines a single training iteration, including the forward pass,
        computation of losses, backpropagation, and weight updates.

        Args:
            data (tensor): Input images.

        Returns:
            dict: Loss values.
        """        
        images_real, class_real = data[0]

        # ToImplement Exercise6a_part3 ==
        # ===============================        
        # Step 1: Train the discriminator

        # Generate images from random latent vectors.
        n_samples = tf.shape(images_real)[0]
        z_random = tf.random.normal((n_samples, self.n_latent_dims))
        class_random = self.generate_random_classes(n_samples)
        z_random_conditional = tf.concat([z_random, class_random], axis=1)
        images_fake = self.generator(z_random_conditional, training=False)
        
        # Create label vectors, 1 for real and 0 for fake images
        labels_real = tf.ones((n_samples, 1))
        labels_fake = tf.zeros((n_samples, 1))

        # Concatenate real and fake images 
        images_disc = tf.concat([images_real, images_fake], axis=0)
        labels_disc = tf.concat([labels_real, labels_fake], axis=0)
       
        with tf.GradientTape() as gt:
            # Predict with the discriminator
            labels_pred, class_pred = self.discriminator(images_disc, training=True)
            
            # Compute discriminator loss for distinguishing real/fake images
            disc_loss_adv = self.loss_bce(labels_disc, labels_pred)
            # Compute discriminator loss for predicting image class, for real images only
            disc_loss_class = self.loss_class(class_real, class_pred[:n_samples, :])
            # Add losses
            disc_loss = disc_loss_adv + disc_loss_class
            
            # Compute classification metric
            self.metric_class.update_state(class_real, class_pred[:n_samples, :])
                                   
        # Compute the gradient of the lost wrt the discriminator weights
        grads_disc = gt.gradient(disc_loss, self.discriminator.trainable_weights)
        
        # Apply the weight updates
        self.optimizer_disc.apply_gradients(zip(grads_disc, self.discriminator.trainable_weights))           
        
        # Step 2: Train the generator
                    
        with tf.GradientTape() as gt2:
            # Generate images from random latent vectors. Generate twice as many
            # images as the batch size so that the generator sees as many
            # samples as the discriminator did. 
            z_random = tf.random.normal((n_samples * 2, self.n_latent_dims))
            class_random = self.generate_random_classes(n_samples * 2)
            z_random_conditional = tf.concat([z_random, class_random], axis=1)
            images_fake = self.generator(z_random_conditional, training=False)
                                    
            # Predict with the discriminator
            labels_pred, class_pred = self.discriminator(images_fake, training=False)
            
            # We want to the discriminator to think these images are real, so we
            # calculate the loss between these predictions and the "real image"
            # labels
            labels_gen = tf.ones((2 * n_samples, 1))
            gen_loss_adv = self.loss_bce(labels_gen, labels_pred)
            # Compute loss between discriminator-predicted classes and the desired classes
            gen_loss_class = self.loss_class(class_random, class_pred)
            # Add losses
            gen_loss = gen_loss_adv + self.cond_loss_weight * gen_loss_class
                        
        # Compute the gradient of the lost wrt the generator weights
        grads_gen = gt2.gradient(gen_loss, self.generator.trainable_weights)
        
        # Apply the weight updates
        self.optimizer_gen.apply_gradients(zip(grads_gen, self.generator.trainable_weights))                
        
        # Update the running means of the losses
        self.loss_gen_tracker.update_state(gen_loss)
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
                'generator_lr': self.generator_lr,
                'discriminator_lr': self.discriminator_lr,
                'n_classes': self.n_classes,
                'cond_loss_weight': self.cond_loss_weight}