
import tensorflow as tf

exp_id = 'final001_002_003_004_005_tf'
saved_model_dir = f'{exp_id}_model'
float16_tflite_path = f'{exp_id}_float16.tflite'

weight_names = [
    'final001_seed0',
    'final002_seed1',
    'final003_seed2',
    'final004_seed3',
    'final005_seed4',
]

ROWS_PER_FRAME = 543  # number of landmarks per frame
LEFT_HAND_INDICES = [468, 469, 470, 471, 472, 473, 474, 475, 476,
                     477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488]
RIGHT_HAND_INDICES = [522, 523, 524, 525, 526, 527, 528, 529, 530,
                      531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542]
OUTER_LIP_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185, 40, 39, 37, 0, 267, 269, 270, 409]
INNER_LIP_INDICES = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 191, 80, 81, 82, 13, 312, 311, 310, 415]
REYE_INDICES = [
    33, 7, 163, 144, 145, 153, 154, 155, 133,
    246, 161, 160, 159, 158, 157, 173,
]
LEYE_INDICES = [
    263, 249, 390, 373, 374, 380, 381, 382, 362,
    466, 388, 387, 386, 385, 384, 398,
]

INDICES = OUTER_LIP_INDICES + INNER_LIP_INDICES + LEFT_HAND_INDICES + RIGHT_HAND_INDICES
EYE_INDICES = OUTER_LIP_INDICES + INNER_LIP_INDICES + LEFT_HAND_INDICES + RIGHT_HAND_INDICES + REYE_INDICES + LEYE_INDICES


def tf_nan_mean(x):
    return tf.reduce_sum(tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)) / tf.reduce_sum(tf.where(tf.math.is_nan(x), tf.zeros_like(x), tf.ones_like(x)))


def tf_nan_std(x):
    return tf.math.sqrt(tf_nan_mean(x * x))


def tf_fillna(x):
    return tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)


class Preprocess(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.max_length = 64
        self.num_features = len(INDICES)

    def call(self, x):
        x = tf.gather(x, INDICES, axis=1)
        x = x - tf_nan_mean(x)
        x = x / tf_nan_std(x)
        x = tf_fillna(x)

        if len(x) > self.max_length:
            x = tf.compat.v1.image.resize(x, size=(self.max_length, self.num_features),
                                          method=tf.image.ResizeMethod.BILINEAR, align_corners=True)

        return tf.reshape(x, (1, -1, self.num_features, 3))


def _preprocess(x):
    features = tf.reshape(x, (-1, len(EYE_INDICES), 3))
    diff_prev_features = features - tf.pad(features, paddings=[[1, 0], [0, 0], [0, 0]], mode='SYMMETRIC')[:-1, :, :]
    diff_next_features = tf.pad(features, paddings=[[0, 1], [0, 0], [0, 0]], mode='SYMMETRIC')[1:, :, :] - features
    velocity_features = (diff_prev_features + diff_next_features) / 2

    motion_features = tf.concat([diff_prev_features, diff_next_features, velocity_features], axis=2)
    motion_features = tf.where(tf.math.is_nan(motion_features), tf.zeros_like(motion_features), motion_features)

    _is_null = tf.cast(features[..., 0] == 0, tf.int32)
    prev_is_null = tf.pad(_is_null, paddings=[[1, 0], [0, 0]])[:-1, :]
    next_is_null = tf.pad(_is_null, paddings=[[0, 1], [0, 0]])[1:, :]
    _is_null = tf.cast(_is_null, bool)
    prev_is_null = tf.cast(prev_is_null, bool)
    next_is_null = tf.cast(next_is_null, bool)
    _is_null = _is_null | prev_is_null | next_is_null
    _is_null = tf.tile(_is_null[..., tf.newaxis], multiples=[1, 1, 9])
    motion_features = tf.where(_is_null, tf.zeros_like(motion_features), motion_features)
    features = features[tf.newaxis, ...]
    motion_features = motion_features[tf.newaxis, ...]
    return features, motion_features


class PreprocessMotionEye(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.max_length48 = 48
        self.max_length64 = 64
        self.num_features = len(EYE_INDICES)

    def call(self, x):
        x = tf.gather(x, EYE_INDICES, axis=1)
        x = x - tf_nan_mean(x)
        x = x / tf_nan_std(x)
        x = tf_fillna(x)

        if len(x) <= self.max_length48:
            x48 = x
            x64 = x
        elif len(x) <= self.max_length64:
            x48 = tf.compat.v1.image.resize(x, size=(self.max_length48, self.num_features),
                                            method=tf.image.ResizeMethod.BILINEAR, align_corners=True)
            x64 = x
        else:
            x48 = tf.compat.v1.image.resize(x, size=(self.max_length48, self.num_features),
                                            method=tf.image.ResizeMethod.BILINEAR, align_corners=True)
            x64 = tf.compat.v1.image.resize(x, size=(self.max_length64, self.num_features),
                                            method=tf.image.ResizeMethod.BILINEAR, align_corners=True)
        features48, motion_features48 = _preprocess(x48)
        features64, motion_features64 = _preprocess(x64)

        return features48, motion_features48, features64, motion_features64


class TFModel(tf.Module):
    def __init__(self):
        super(TFModel, self).__init__()
        self.preprocess = Preprocess()
        self.preprocess_motion_eye = PreprocessMotionEye()
        self.preprocess.trainable = False
        self.preprocess_motion_eye.trainable = False

        self.net0 = tf.saved_model.load(f'output_models/{weight_names[0]}')
        self.net1 = tf.saved_model.load(f'output_models/{weight_names[1]}')
        self.net2 = tf.saved_model.load(f'output_models/{weight_names[2]}')
        self.net3 = tf.saved_model.load(f'output_models/{weight_names[3]}')
        self.net4 = tf.saved_model.load(f'output_models/{weight_names[4]}')
        self.net0.trainable = False
        self.net1.trainable = False
        self.net2.trainable = False
        self.net3.trainable = False
        self.net4.trainable = False

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, 543, 3], dtype=tf.float32, name='inputs')
    ])
    def call(self, x):
        features = self.preprocess(x)
        features48, motion_features48, features64, motion_features64 = self.preprocess_motion_eye(x)

        logits = tf.reduce_mean(tf.stack([
            self.net0(features),
            self.net1(features48, motion_features48),
            self.net2(features),
            self.net3(features64, motion_features64),
            self.net4(features48, motion_features48),
        ], axis=0), axis=0)
        outputs = {'outputs': logits}
        return outputs


def convert(saved_model_dir: str, save_path: str):
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tf_lite_model = converter.convert()
    with open(save_path, 'wb') as f:
        f.write(tf_lite_model)


def main():
    tf_model = TFModel()
    tf.saved_model.save(tf_model, saved_model_dir)
    convert(saved_model_dir, float16_tflite_path)


if __name__ == '__main__':
    main()
