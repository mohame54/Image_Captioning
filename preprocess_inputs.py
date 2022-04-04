import tensorflow as tf
from collections import Counter
from keras import layers


def load_img(img_path, size=(299, 299)):
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, size)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image


def standerize(inputs):
    inputs = tf.strings.lower(inputs)
    return tf.strings.regex_replace(inputs,
                                    r"!\"#$%&\(\)\*\+.,-/:;=?@\[\\\]^_`{|}~", "")


def make_ds(features, labels, batch_size,
            vactor=None):
    def tokenize(x):
        return vactor(x)

    feats = tf.data.Dataset.from_tensor_slices(features).map(load_img)
    labels = tf.data.Dataset.from_tensor_slices(labels).map(tokenize)
    return tf.data.Dataset.zip((feats, labels)).shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)


class CaptionProcessor:
    def __init__(self, caption_dict: dict, num_samples):
        self.cap_dict = caption_dict
        self.img_paths = list(caption_dict.keys())[:num_samples]
        self.initial_vocab = Counter()

    def get_features_labels(self):
        features, labels = [], []
        max_len = 0
        for path in self.img_paths:
            caps = self.cap_dict[path]
            for cap in caps:
                cap = cap.split()
                self.initial_vocab.update(cap)
                if len(cap) > max_len:
                    max_len = len(cap)
            features.extend([path] * len(caps))
            labels.extend(caps)
        return features, labels, max_len

    def get_tokenizer(self):
        _, captions, max_len = self.get_features_labels()
        captions = tf.data.Dataset.from_tensor_slices(captions)
        tokenizer = layers.TextVectorization(max_tokens=len(self.initial_vocab),
                                             output_sequence_length=max_len, standardize=standerize)
        tokenizer.adapt(captions)
        return  tokenizer


