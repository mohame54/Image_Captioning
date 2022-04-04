import tensorflow as tf
from keras import layers


class TransformerEncoder(layers.Layer):
    def __init__(self, num_heads, embed_dim, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.attn_layer = layers.MultiHeadAttention(num_heads, embed_dim)
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.dense = layers.Dense(embed_dim, activation='relu')

    def call(self, inputs, training=True, mask=None):
        out = self.norm1(inputs)
        out = self.dense(out)
        attn_output = self.attn_layer(query=out, key=out, value=out,
                                      attention_mask=mask, training=training)
        return self.norm2(attn_output + out)


class PositionalEncoding(layers.Layer):
    def __init__(self, embed_dim, seq_len, vocab_size, **kwargs):
        super(PositionalEncoding, self).__init__(name='PosEn')
        self.embed_token = layers.Embedding(vocab_size, embed_dim)
        self.embed_seq = layers.Embedding(seq_len, embed_dim)
        self.embed_scale = tf.math.sqrt(tf.cast(embed_dim, tf.float32))

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        position = tf.range(0, length, delta=1)
        embed_tokens = self.embed_token(inputs) * self.embed_scale
        embed_pos = self.embed_seq(position)
        return embed_tokens + embed_pos

    def compute_mask(self, inputs, mask):
        return tf.math.not_equal(inputs, 0)


class TransformerDecoder(layers.Layer):
    def __init__(self, vocab_size,
                 ffd, embed_dim, num_heads, rate, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.attn_layer1 = layers.MultiHeadAttention(num_heads, embed_dim,
                                                     dropout=0.1)
        self.attn_layer2 = layers.MultiHeadAttention(num_heads, embed_dim,
                                                     dropout=0.1)
        self.ffd1 = layers.Dense(ffd, activation='relu')
        self.ffd2 = layers.Dense(embed_dim)

        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.norm3 = layers.LayerNormalization()

        self.drop1 = layers.Dropout(rate)
        self.drop2 = layers.Dropout(0.5)

        self.final = layers.Dense(vocab_size, activation='softmax')
        self.embed = PositionalEncoding(embed_dim, 47, vocab_size)
        self.supports_masking = True

    def casual_mask(self, inputs):
        shape = tf.shape(inputs)
        batch_size, seq_len = shape[0], shape[1]
        i = tf.range(seq_len)[:, tf.newaxis]
        j = tf.range(seq_len)
        mask = tf.cast(i >= j, dtype='int32')
        mask = tf.reshape(mask, (1, shape[1], shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
            axis=0,
        )
        return tf.tile(mask, mult)

    def call(self, inputs, enc_out,
             training=True, mask=None):
        inputs = self.embed(inputs)
        casual_mask = self.casual_mask(inputs)
        if mask is not None:
            padding_mask = tf.cast(mask[:, :, tf.newaxis], dtype=tf.int32)
            combined_mask = tf.cast(mask[:, tf.newaxis, :], dtype=tf.int32)
            combined_mask = tf.minimum(combined_mask, casual_mask)
        attn1 = self.attn_layer1(
            key=inputs,
            value=inputs,
            query=inputs,
            attention_mask=combined_mask,
            training=training
        )
        out1 = self.norm1(inputs + attn1)
        att2 = self.attn_layer2(
            query=out1,
            key=enc_out,
            value=enc_out,
            attention_mask=padding_mask,
            training=training
        )
        out2 = self.norm2(att2 + out1)
        ffn_output = self.ffd1(out2)
        ffn_output = self.drop1(ffn_output, training=training)

        ffn_output = self.norm3(ffn_output + out2, training=training)
        ffn_out = self.drop2(ffn_output, training=training)
        return self.final(ffn_out)
