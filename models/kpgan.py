import tensorflow as tf

def gradient_penality(critic, fake_samples, real_samples, labels):
    batch_size = fake_samples.shape[0]
    alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
    inter_sample = real_samples * alpha + fake_samples * (1.0 - alpha)

    with tf.GradientTape() as tape_gp:
        tape_gp.watch(inter_sample)
        inter_score = critic((inter_sample, labels))

    gp_gradients = tape_gp.gradient(inter_score, inter_sample)
    gp_gradients_norm = tf.sqrt(tf.reduce_sum(tf.square(gp_gradients), axis=[1, 2, 3]))
    return tf.reduce_mean((gp_gradients_norm - 1.0) ** 2)

def generator_loss(fake_score):
    return -tf.reduce_mean(fake_score)

def critic_loss(fake_score, real_score, gp, gp_lambda):
    return tf.reduce_mean(fake_score) - tf.reduce_mean(real_score) + gp * gp_lambda
    
class Generator10(tf.keras.Model):
  def __init__(self, classes):
    super(Generator10, self).__init__()
    # Image generation input
    self.dense1 = tf.keras.layers.Dense(1*2*256)
    self.reshape1 = tf.keras.layers.Reshape((1, 2, 256))

    # Label embedding input
    self.embed = tf.keras.layers.Embedding(classes, 10)
    self.dense2 = tf.keras.layers.Dense(1*2)
    self.reshape2 = tf.keras.layers.Reshape((1, 2, classes))

    # Convolutional upsampling layers
    self.conv1 = tf.keras.layers.Conv2DTranspose(256, (3, 3), strides=2, padding='same', use_bias=True)
    self.conv2 = tf.keras.layers.Conv2DTranspose(128, (3, 2), strides=2, padding='valid', use_bias=True)
    self.conv3 = tf.keras.layers.Conv2DTranspose( 64, (3, 3), strides=1, padding='same', use_bias=True)
    self.conv4 = tf.keras.layers.Conv2DTranspose(  3, (2, 3), strides=2, padding='valid', use_bias=True)
    
  def call(self, inputs):
    x = inputs[0]
    label = inputs[1]

    x = tf.nn.leaky_relu(self.dense1(x))
    x = self.reshape1(x)

    label = self.embed(label)
    label = self.dense2(label)
    label = self.reshape2(label)

    x = tf.concat([x, label], -1)

    x = tf.nn.leaky_relu(self.conv1(x))
    x = tf.nn.leaky_relu(self.conv2(x))
    x = tf.nn.leaky_relu(self.conv3(x))
    x = tf.nn.tanh(self.conv4(x))

    return x

class Generator20(tf.keras.Model):
  def __init__(self, classes):
    super(Generator20, self).__init__()
    # Image generation input
    self.dense1 = tf.keras.layers.Dense(1*2*256)
    self.reshape1 = tf.keras.layers.Reshape((1, 2, 256))

    # Label embedding input
    self.embed = tf.keras.layers.Embedding(classes, 10)
    self.dense2 = tf.keras.layers.Dense(1*2)
    self.reshape2 = tf.keras.layers.Reshape((1, 2, classes))

    # Convolutional upsampling layers
    self.conv1 = tf.keras.layers.Conv2DTranspose(256, (3, 3), strides=2, padding='same', use_bias=True)
    self.conv2 = tf.keras.layers.Conv2DTranspose(128, (3, 2), strides=2, padding='valid', use_bias=True)
    self.conv3 = tf.keras.layers.Conv2DTranspose( 64, (3, 3), strides=(2, 1), padding='same', use_bias=True)
    self.conv4 = tf.keras.layers.Conv2DTranspose(  3, (2, 3), strides=2, padding='valid', use_bias=True)
    
  def call(self, inputs):
    x = inputs[0]
    label = inputs[1]

    x = tf.nn.leaky_relu(self.dense1(x))
    x = self.reshape1(x)

    label = self.embed(label)
    label = self.dense2(label)
    label = self.reshape2(label)

    x = tf.concat([x, label], -1)

    x = tf.nn.leaky_relu(self.conv1(x))
    x = tf.nn.leaky_relu(self.conv2(x))
    x = tf.nn.leaky_relu(self.conv3(x))
    x = tf.nn.tanh(self.conv4(x))

    return x

class Generator60(tf.keras.Model):
  def __init__(self, classes):
    super(Generator60, self).__init__()
    # Image generation input
    self.dense1 = tf.keras.layers.Dense(7*2*256)
    self.reshape1 = tf.keras.layers.Reshape((7, 2, 256))

    # Label embedding input
    self.embed = tf.keras.layers.Embedding(classes, 10)
    self.dense2 = tf.keras.layers.Dense(7*2)
    self.reshape2 = tf.keras.layers.Reshape((7, 2, classes))

    # Convolutional upsampling layers
    self.conv1 = tf.keras.layers.Conv2DTranspose(256, (3, 2), strides=2, padding='valid', use_bias=True)
    self.conv2 = tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=2, padding='same', use_bias=True)
    self.conv3 = tf.keras.layers.Conv2DTranspose( 64, (3, 3), strides=1, padding='same', use_bias=True)
    self.conv4 = tf.keras.layers.Conv2DTranspose(  3, (2, 3), strides=2, padding='valid', use_bias=True)
    
  def call(self, inputs):
    x = inputs[0]
    label = inputs[1]

    x = tf.nn.leaky_relu(self.dense1(x))
    x = self.reshape1(x)

    label = self.embed(label)
    label = self.dense2(label)
    label = self.reshape2(label)

    x = tf.concat([x, label], -1)

    x = tf.nn.leaky_relu(self.conv1(x))
    x = tf.nn.leaky_relu(self.conv2(x))
    x = tf.nn.leaky_relu(self.conv3(x))
    x = tf.nn.tanh(self.conv4(x))

    return x

class Critic(tf.keras.Model):
  def __init__(self, classes, number_frames):
    super(Critic, self).__init__()
    self.embed = tf.keras.layers.Embedding(classes, 10)
    self.dense2 = tf.keras.layers.Dense(number_frames*17)
    self.reshape2 = tf.keras.layers.Reshape((number_frames, 17, classes))

    self.conv1 = tf.keras.layers.Conv2D(32, (5, 5), strides=2, padding='same')
    self.conv2 = tf.keras.layers.Conv2D(64, (5, 5), strides=2, padding='same')
    self.conv3 = tf.keras.layers.Conv2D(128, (5, 5), strides=2, padding='same')

    self.flatten = tf.keras.layers.Flatten()
    self.dense = tf.keras.layers.Dense(1)

  def call(self, inputs):
    x = inputs[0]
    label = inputs[1]

    label = self.embed(label)
    label = self.dense2(label)
    label = self.reshape2(label)

    x = tf.concat([x, label], -1)

    x = tf.nn.dropout(tf.nn.leaky_relu(self.conv1(x), 0.3), 0.3)
    x = tf.nn.dropout(tf.nn.leaky_relu(self.conv2(x), 0.3), 0.3)
    x = tf.nn.dropout(tf.nn.leaky_relu(self.conv3(x), 0.3), 0.3)

    x = self.dense(self.flatten(x))
    return x