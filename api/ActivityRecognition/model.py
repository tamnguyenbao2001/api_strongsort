import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow_hub as hub



EMBED_DIM = 4
NUM_HEADS = 2
FF_DIM = 16
VOCAB_SIZE = 1030
MAXLEN = 255
WEIGHT_DIR = "ActivityRecognition/ActivityRecognizer/my_checkpoint"



class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

def transformer():
    inputs = keras.Input(shape=(MAXLEN,))
    embedding_layer = TokenAndPositionEmbedding(MAXLEN, VOCAB_SIZE, EMBED_DIM)
    scale_layer = tf.keras.layers.Lambda(lambda x: x*1000)
    x = scale_layer(inputs)
    x = embedding_layer(x)
    transformer_block = TransformerBlock(EMBED_DIM, NUM_HEADS, FF_DIM)
    x = transformer_block(x)
    x = keras.layers.Flatten()(x)
    outputs = layers.Dense(3)(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.load_weights(WEIGHT_DIR)
    model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
             optimizer = keras.optimizers.Adam(learning_rate=1e-4), metrics=["acc"])

    return keras.Model(inputs=inputs, outputs=outputs) 


