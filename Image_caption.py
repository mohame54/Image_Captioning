import tensorflow as tf
from Layers.Transformer_layers import *


class ImageCaption(tf.keras.Model):
    def __init__(self, cnn_model, num_heads,
                 embed_dim, ffd, rate, aug_model, vocab
                 , **kwargs):

        super(ImageCaption, self).__init__(**kwargs)
        self.cnn_model = cnn_model
        self.encoder = TransformerEncoder(num_heads, embed_dim, name='Encoder')
        self.decoder = TransformerDecoder(len(vocab),
                                          ffd, embed_dim, num_heads, rate, name='Decoder')
        self.aug = aug_model

    @tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.string, name='path')])
    def preprocess_load_img(self, img_path, mode='jpg'):
        img = tf.io.read_file(img_path)
        if mode == 'jpg':
            img = tf.image.decode_jpeg(img, channels=3)
        else:
            img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, (299, 299))
        img = tf.image.convert_image_dtype(img, tf.float32)
        return img[None]

    def cal_loss(self, y_true, y_pred, mask):
        loss = self.loss(y_true, y_pred)
        mask = tf.cast(mask, loss.dtype)
        loss *= mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)

    def cal_acc(self, y_true, y_pred, mask):
        accuracy = tf.equal(y_true, tf.argmax(y_pred, axis=2))
        accs = tf.math.logical_and(mask, accuracy)
        accs = tf.cast(accs, tf.float32)
        mask = tf.cast(mask, tf.float32)
        return tf.reduce_sum(accs) / tf.reduce_sum(mask)

    def _compute_caption_loss_and_acc(self, img_embed,
                                      batch_seq, training=False):

        encoder_out = self.encoder(img_embed, training=training)
        batch_seq_inp = batch_seq[:, :-1]
        batch_seq_true = batch_seq[:, 1:]
        mask = tf.math.not_equal(batch_seq_true, 0)
        batch_seq_pred = self.decoder(
            batch_seq_inp, encoder_out, training=training, mask=mask
        )
        loss = self.cal_loss(batch_seq_true, batch_seq_pred, mask)
        acc = self.cal_acc(batch_seq_true, batch_seq_pred, mask)
        return loss, acc

    @tf.function
    def train_step(self, batch_data):
        batch_img, batch_seq = batch_data
        if tf.random.uniform(()) > 0.5:
            batch_img = self.aug(batch_img)
        img_embed = self.cnn_model(batch_img)
        with tf.GradientTape() as tape:
            loss, acc = self._compute_caption_loss_and_acc(img_embed, batch_seq, training=True)
        train_vars = self.decoder.trainable_variables + self.encoder.trainable_variables
        grads = tape.gradient(loss, train_vars)
        self.optimizer.apply_gradients(zip(grads, train_vars))
        return loss, acc

    @tf.function
    def test_step(self, batch_data):
        batch_img, batch_seq = batch_data
        img_embed = self.cnn_model(batch_img)
        loss, acc = self._compute_caption_loss_and_acc(img_embed, batch_seq)
        return loss, acc

    def evaluate(self, test_dataset):
        total_loss, total_accs = 0, 0
        for batch in test_dataset:
            loss, acc = self.test_step(batch)
            total_loss += loss
            total_accs += acc
        total_losses = total_loss / len(test_dataset)
        total_accs = total_accs / len(test_dataset)
        return total_losses, total_accs


def get_aug_model():
    return tf.keras.Sequential([
        layers.RandomFlip('horizontal'),
        layers.RandomContrast(0.3),
        layers.RandomRotation(0.2)])


def get_cnn_model():
    base_model = tf.keras.applications.EfficientNetB0(include_top=False)
    base_model.trainable = False
    inputs = base_model.inputs
    output = layers.Reshape((-1, base_model.output.shape[-1]))(base_model.output)
    return tf.keras.Model(inputs, output)


class LRSchedule(tf.optimizers.schedules.LearningRateSchedule):
    def __init__(self, post_warmup_learning_rate, warmup_steps):
        super().__init__()
        self.post_warmup_learning_rate = post_warmup_learning_rate
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        global_step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        warmup_progress = global_step / warmup_steps
        warmup_learning_rate = self.post_warmup_learning_rate * warmup_progress
        return tf.cond(
            global_step < warmup_steps,
            lambda: warmup_learning_rate,
            lambda: self.post_warmup_learning_rate,
        )
