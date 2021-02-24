# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Efficientnet based feature extractors for CenterNet[1] meta architecture.

[1]: https://arxiv.org/abs/1904.07850
"""


import tensorflow.compat.v1 as tf

from object_detection.meta_architectures.center_net_meta_arch import CenterNetFeatureExtractor


class CenterNetEfficientnetFeatureExtractor(CenterNetFeatureExtractor):
    """Efficientnet base feature extractor for the CenterNet model."""

    def __init__(self, efficientnet_type, channel_means=(0., 0., 0.),
                 channel_stds=(1., 1., 1.), bgr_ordering=False):
        """Initializes the feature extractor with a specific EfficientNet architecture.

        Args:
          efficientnet_type: A string specifying which kind of EfficientNet to use. Currently
            only `efficientnet_b0` to `efficientnet_b7` are supported.
          channel_means: A tuple of floats, denoting the mean of each channel
            which will be subtracted from it.
          channel_stds: A tuple of floats, denoting the standard deviation of each
            channel. Each channel will be divided by its standard deviation value.
          bgr_ordering: bool, if set will change the channel ordering to be in the
            [blue, red, green] order.

        """

        super(CenterNetEfficientnetFeatureExtractor, self).__init__(
            channel_means=channel_means, channel_stds=channel_stds,
            bgr_ordering=bgr_ordering)
        if efficientnet_type == 'efficientnet_b0':
            self._base_model = tf.keras.applications.EfficientNetB0(weights=None,
                                                                    include_top=False)
        elif efficientnet_type == 'efficientnet_b1':
            self._base_model = tf.keras.applications.EfficientNetB1(weights=None,
                                                                    include_top=False)
        elif efficientnet_type == 'efficientnet_b2':
            self._base_model = tf.keras.applications.EfficientNetB2(weights=None,
                                                                    include_top=False)
        elif efficientnet_type == 'efficientnet_b3':
            self._base_model = tf.keras.applications.EfficientNetB3(weights=None,
                                                                    include_top=False)
        elif efficientnet_type == 'efficientnet_b4':
            self._base_model = tf.keras.applications.EfficientNetB4(weights=None,
                                                                    include_top=False)
        elif efficientnet_type == 'efficientnet_b5':
            self._base_model = tf.keras.applications.EfficientNetB5(weights=None,
                                                                    include_top=False)
        elif efficientnet_type == 'efficientnet_b6':
            self._base_model = tf.keras.applications.EfficientNetB6(weights=None,
                                                                    include_top=False)
        elif efficientnet_type == 'efficientnet_b7':
            self._base_model = tf.keras.applications.EfficientNetB7(weights=None,
                                                                    include_top=False)
        else:
            raise ValueError(
                'Unknown EfficientNet Model {}'.format(efficientnet_type))
        output_layer = self._base_model.get_layer('top_conv')

        self._efficientnet_model = tf.keras.models.Model(inputs=self._base_model.input,
                                                         outputs=output_layer.output)
        efficientnet_output = self._efficientnet_model(self._base_model.input)

        for num_filters in [256, 128, 64]:
            # TODO This section has a few differences from the paper
            # Figure out how much of a performance impact they have.

            # 1. We use a simple convolution instead of a deformable convolution
            conv = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=3,
                                          strides=1, padding='same')
            efficientnet_output = conv(efficientnet_output)
            efficientnet_output = tf.keras.layers.BatchNormalization()(efficientnet_output)
            efficientnet_output = tf.keras.layers.ReLU()(efficientnet_output)

            # 2. We use the default initialization for the convolution layers
            # instead of initializing it to do bilinear upsampling.
            conv_transpose = tf.keras.layers.Conv2DTranspose(filters=num_filters,
                                                             kernel_size=3, strides=2,
                                                             padding='same')
            efficientnet_output = conv_transpose(efficientnet_output)
            efficientnet_output = tf.keras.layers.BatchNormalization()(efficientnet_output)
            efficientnet_output = tf.keras.layers.ReLU()(efficientnet_output)

        self._feature_extractor_model = tf.keras.models.Model(
            inputs=self._base_model.input, outputs=efficientnet_output)

    def preprocess(self, resized_inputs):
        """Preprocess input images for the EfficientNet model.

        This scales images in the range [0, 255] to the range [-1, 1]

        Args:
          resized_inputs: a [batch, height, width, channels] float32 tensor.

        Returns:
          outputs: a [batch, height, width, channels] float32 tensor.

        """
        resized_inputs = super(CenterNetEfficientnetFeatureExtractor, self).preprocess(
            resized_inputs)
        return tf.keras.applications.efficientnet.preprocess_input(resized_inputs)

    def load_feature_extractor_weights(self, path):
        self._base_model.load_weights(path)

    def call(self, inputs):
        """Returns image features extracted by the backbone.

        Args:
          inputs: An image tensor of shape [batch_size, input_height,
            input_width, 3]

        Returns:
          features_list: A list of length 1 containing a tensor of shape
            [batch_size, input_height // 4, input_width // 4, 64] containing
            the features extracted by the EfficientNet.
        """
        return [self._feature_extractor_model(inputs)]

    @property
    def num_feature_outputs(self):
        return 1

    @property
    def out_stride(self):
        return 4

    @property
    def supported_sub_model_types(self):
        return ['classification']

    def get_sub_model(self, sub_model_type):
        if sub_model_type == 'classification':
            return self._base_model
        else:
            ValueError(
                'Sub model type "{}" not supported.'.format(sub_model_type))


def efficientnet_b0(channel_means, channel_stds, bgr_ordering):
    """The EfficientNet B0 feature extractor."""

    return CenterNetEfficientnetFeatureExtractor(
        efficientnet_type='efficientnet_b0',
        channel_means=channel_means,
        channel_stds=channel_stds,
        bgr_ordering=bgr_ordering
    )


def efficientnet_b1(channel_means, channel_stds, bgr_ordering):
    """The EfficientNet B1 feature extractor."""

    return CenterNetEfficientnetFeatureExtractor(
        efficientnet_type='efficientnet_b1',
        channel_means=channel_means,
        channel_stds=channel_stds,
        bgr_ordering=bgr_ordering)


def efficientnet_b2(channel_means, channel_stds, bgr_ordering):
    """The EfficientNet B2 feature extractor."""

    return CenterNetEfficientnetFeatureExtractor(
        efficientnet_type='efficientnet_b2',
        channel_means=channel_means,
        channel_stds=channel_stds,
        bgr_ordering=bgr_ordering)


def efficientnet_b3(channel_means, channel_stds, bgr_ordering):
    """The EfficientNet B3 feature extractor."""

    return CenterNetEfficientnetFeatureExtractor(
        efficientnet_type='efficientnet_b3',
        channel_means=channel_means,
        channel_stds=channel_stds,
        bgr_ordering=bgr_ordering)


def efficientnet_b4(channel_means, channel_stds, bgr_ordering):
    """The EfficientNet B4 feature extractor."""

    return CenterNetEfficientnetFeatureExtractor(
        efficientnet_type='efficientnet_b4',
        channel_means=channel_means,
        channel_stds=channel_stds,
        bgr_ordering=bgr_ordering)


def efficientnet_b5(channel_means, channel_stds, bgr_ordering):
    """The EfficientNet B5 feature extractor."""

    return CenterNetEfficientnetFeatureExtractor(
        efficientnet_type='efficientnet_b5',
        channel_means=channel_means,
        channel_stds=channel_stds,
        bgr_ordering=bgr_ordering)


def efficientnet_b6(channel_means, channel_stds, bgr_ordering):
    """The EfficientNet B6 feature extractor."""

    return CenterNetEfficientnetFeatureExtractor(
        efficientnet_type='efficientnet_b6',
        channel_means=channel_means,
        channel_stds=channel_stds,
        bgr_ordering=bgr_ordering)


def efficientnet_b7(channel_means, channel_stds, bgr_ordering):
    """The EfficientNet B7 feature extractor."""

    return CenterNetEfficientnetFeatureExtractor(
        efficientnet_type='efficientnet_b7',
        channel_means=channel_means,
        channel_stds=channel_stds,
        bgr_ordering=bgr_ordering)
