import tensorflow as tf


class CaptionMaker(tf.Module):
    def __init__(self, vocab, tokenizer, cnn_model,
                 decoder, encoder, max_len, **kwargs):
        super(CaptionMaker, self).__init__(**kwargs)
        self.tokenizer = tokenizer
        self.vocab = tokenizer.get_vocabulary()
        self.decoder = decoder
        self.encoder = encoder
        self.feat_ex = cnn_model
        self.vocab = vocab
        self.max_len = max_len

    @tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.string, name='path'),
                                  tf.TensorSpec(shape=(), dtype=tf.string, name='type')])
    def preprocess_load_img(self, img_path, mode='jpg'):
        img = tf.io.read_file(img_path)
        if mode == 'jpg':
            img = tf.image.decode_jpeg(img, channels=3)
        else:
            img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, (299, 299))
        img = tf.image.convert_image_dtype(img, tf.float32)
        return img[None]

    @tf.function(input_signature=[tf.TensorSpec(dtype=tf.string, shape=(), name='Text')])
    def tokenize(self, text):
        return self.tokenizer(text)

    @tf.function(input_signature=[tf.TensorSpec(dtype=tf.int64, shape=(), name='index')])
    def get_word(self, idx):
        return tf.gather(self.vocab, idx)

    @tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.string, name='path'),
                                  tf.TensorSpec(shape=(), dtype=tf.string, name='type')])
    def make_caption(self, path, mode):
        img = self.preprocess_load_img(path, mode=mode)
        img_embed = self.feat_ex(img)
        enc_out = self.encoder(img_embed, training=False)
        stc = tf.constant('<start> ')
        for i in tf.range(self.max_len - 1):
            dec_inputs = tf.expand_dims(self.tokenize(stc), 0)[:, :-1]
            mask = tf.not_equal(dec_inputs, 0)
            preds = self.decoder(dec_inputs, enc_out,
                                 training=False, mask=mask)
            token_idx = tf.argmax(preds[0, i, :])
            word = self.get_word(token_idx)
            if word == tf.constant('<end>'):
                break
            stc = tf.add(stc, ' ')
            stc = tf.add(stc, word)
        stc = tf.strings.regex_replace(stc, '<start> ', '')
        stc = tf.strings.regex_replace(stc, ' <end>', '')
        return stc
