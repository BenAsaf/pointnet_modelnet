import tensorflow as tf
import numpy as np
from tensorflow.keras import layers


class InputTransformNet(tf.keras.Model):

    def get_config(self):
        return {
            "num_features": self._num_features,
            "bn_momentum": self._bn_momentum
        }

    def __init__(self, num_features: int, bn_momentum: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._num_features = num_features
        self._bn_momentum = bn_momentum

        self._eye = tf.constant(np.reshape(np.eye(self._num_features), (-1, )), dtype=tf.float32)

        self.conv1 = layers.Conv1D(filters=64, kernel_size=1, padding="valid")
        self.conv1_bn = layers.BatchNormalization(momentum=self._bn_momentum)

        self.conv2 = layers.Conv1D(filters=128, kernel_size=1, padding="valid")
        self.conv2_bn = layers.BatchNormalization(momentum=self._bn_momentum)

        self.conv3 = layers.Conv1D(filters=1024, kernel_size=1, padding="valid")
        self.conv3_bn = layers.BatchNormalization(momentum=self._bn_momentum)

        self.pool = layers.GlobalMaxPooling1D()

        self.fc1 = layers.Dense(units=512)
        self.fc1_bn = layers.BatchNormalization(momentum=self._bn_momentum)

        self.fc2 = layers.Dense(units=256)
        self.fc2_bn = layers.BatchNormalization(momentum=self._bn_momentum)

        self.fc3 = layers.Dense(units=self._num_features * self._num_features,
                                kernel_initializer="zeros",
                                bias_initializer="zeros")

        self.reshape = layers.Reshape((self._num_features, self._num_features))

    def call(self, inputs, training=None, mask=None):
        x = inputs

        x = tf.nn.swish(self.conv1_bn(self.conv1(x), training=training))

        x = tf.nn.swish(self.conv2_bn(self.conv2(x), training=training))

        x = tf.nn.swish(self.conv3_bn(self.conv3(x), training=training))

        x = self.pool(x)

        x = tf.nn.swish(self.fc1_bn(self.fc1(x), training=training))

        x = tf.nn.swish(self.fc2_bn(self.fc2(x), training=training))

        x = self.fc3(x)
        x = x + self._eye

        x = self.reshape(x)

        return x


class FeatureTransformNet(tf.keras.Model):

    def get_config(self):
        return {
            "num_features": self._num_features,
            "bn_momentum": self._bn_momentum
        }

    def __init__(self, num_features: int, bn_momentum: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._num_features = num_features
        self._bn_momentum = bn_momentum

        self._eye = tf.constant(np.reshape(np.eye(self._num_features), (-1, )), dtype=tf.float32)

        self.conv1 = layers.Conv1D(filters=64, kernel_size=1, padding="valid")
        self.conv1_bn = layers.BatchNormalization(momentum=self._bn_momentum)

        self.conv2 = layers.Conv1D(filters=128, kernel_size=1, padding="valid")
        self.conv2_bn = layers.BatchNormalization(momentum=self._bn_momentum)

        self.conv3 = layers.Conv1D(filters=1024, kernel_size=1, padding="valid")
        self.conv3_bn = layers.BatchNormalization(momentum=self._bn_momentum)

        self.pool = layers.GlobalMaxPooling1D()

        self.fc1 = layers.Dense(units=512)
        self.fc1_bn = layers.BatchNormalization(momentum=self._bn_momentum)

        self.fc2 = layers.Dense(units=256)
        self.fc2_bn = layers.BatchNormalization(momentum=self._bn_momentum)

        self.fc3 = layers.Dense(units=num_features * num_features,
                                kernel_initializer="zeros",
                                bias_initializer="zeros")

        self.reshape = layers.Reshape((self._num_features, self._num_features))

    def call(self, inputs, training=None, mask=None):

        x = tf.nn.swish(self.conv1_bn(self.conv1(inputs), training=training))

        x = tf.nn.swish(self.conv2_bn(self.conv2(x), training=training))

        x = tf.nn.swish(self.conv3_bn(self.conv3(x), training=training))

        x = self.pool(x)

        x = tf.nn.swish(self.fc1_bn(self.fc1(x), training=training))

        x = tf.nn.swish(self.fc2_bn(self.fc2(x), training=training))

        x = self.fc3(x)
        x = x + self._eye

        x = self.reshape(x)

        return x


class PointNetModel(tf.keras.Model):

    def get_config(self):
        return {
            "num_classes": self._num_classes
        }

    def __init__(self, num_classes: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._momentum = 0.0
        self._num_classes = num_classes

        self.tnet0 = InputTransformNet(num_features=3, bn_momentum=self._momentum)
        self.tnet0_dot = layers.Dot(axes=(2, 1))

        self.conv1 = layers.Conv1D(filters=64, kernel_size=3, padding="valid")
        self.bn1 = layers.BatchNormalization(momentum=self._momentum)

        self.conv2 = layers.Conv1D(filters=64, kernel_size=1, padding="valid")
        self.bn2 = layers.BatchNormalization(momentum=self._momentum)

        self.tnet3 = FeatureTransformNet(num_features=64, bn_momentum=self._momentum)
        self.tnet3_dot = layers.Dot(axes=(2, 1))

        self.conv4 = layers.Conv1D(filters=64, kernel_size=1, padding="valid")
        self.conv4_bn = layers.BatchNormalization(momentum=self._momentum)

        self.conv5 = layers.Conv1D(filters=128, kernel_size=1, padding="valid")
        self.conv5_bn = layers.BatchNormalization(momentum=self._momentum)

        self.conv6 = layers.Conv1D(filters=1024, kernel_size=1, padding="valid")
        self.conv6_bn = layers.BatchNormalization(momentum=self._momentum)

        self.pool = layers.GlobalMaxPooling1D()

        self.fc1 = layers.Dense(units=1024)
        self.fc1_bn = layers.BatchNormalization(momentum=self._momentum)
        self.fc1_dropout = layers.Dropout(0.3)

        self.fc2 = layers.Dense(units=512)
        self.fc2_bn = layers.BatchNormalization(momentum=self._momentum)
        self.fc2_dropout = layers.Dropout(0.3)

        self.fc3 = layers.Dense(units=128)
        self.fc3_bn = layers.BatchNormalization(momentum=self._momentum)

        self.predict = layers.Dense(units=num_classes)

    @tf.function
    def call(self, inputs, training=None, mask=None):
        points = inputs

        x_transformed = self.tnet0(points, training=training, mask=mask)
        x = self.tnet0_dot([points, x_transformed])

        x = tf.nn.swish(self.bn1(self.conv1(x), training=training))
        x = tf.nn.swish(self.bn2(self.conv2(x), training=training))
        x_transformed = self.tnet3(x, training=training, mask=mask)
        x = self.tnet3_dot([x, x_transformed])

        x = tf.nn.swish(self.conv4_bn(self.conv4(x), training=training))
        x = tf.nn.swish(self.conv5_bn(self.conv5(x), training=training))
        x = tf.nn.swish(self.conv6_bn(self.conv6(x), training=training))

        x = self.pool(x)

        x = tf.nn.swish(self.fc1_bn(self.fc1(x), training=training))
        x = self.fc1_dropout(x, training=training)
        x = tf.nn.swish(self.fc2_bn(self.fc2(x), training=training))
        x = self.fc2_dropout(x, training=training)

        output = tf.nn.swish(self.fc3_bn(self.fc3(x), training=training))
        output = self.predict(output)
        return output
