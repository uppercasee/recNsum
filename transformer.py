import numpy as np
import pandas as pd
import tensorflow as tf
import re
import pickle
from model import Transformer, optimizer

ENCODER_LEN = 100
DECODER_LEN = 20
BATCH_SIZE = 64
BUFFER_SIZE = BATCH_SIZE * 8
num_layers = 3
d_model = 128
dff = 512
num_heads = 4
dropout_rate = 0.2
EPOCHS = 30

news = pd.read_excel("datasets/Inshorts Cleaned Data.xlsx", engine="openpyxl")
news.drop(["Source ", "Time ", "Publish Date"], axis=1, inplace=True)
print("dataset loaded successfully.")

article = news["Short"]
summary = news["Headline"]
article = article.apply(lambda x: "<SOS> " + x + " <EOS>")
summary = summary.apply(lambda x: "<SOS> " + x + " <EOS>")


def preprocess(text):
    text = re.sub(r"&.[1-9]+;", " ", text)
    return text


# Load article_tokenizer
with open("summary_datas/article_tokenizer.pkl", "rb") as f:
    article_tokenizer = pickle.load(f)
    print("article tokenizer loaded successfully.")

# Load summary_tokenizer
with open("summary_datas/summary_tokenizer.pkl", "rb") as f:
    summary_tokenizer = pickle.load(f)
    print("summary tokenizer loaded successfully.")

ENCODER_VOCAB = len(article_tokenizer.word_index) + 1
DECODER_VOCAB = len(summary_tokenizer.word_index) + 1

transformer = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    input_vocab_size=ENCODER_VOCAB,
    target_vocab_size=DECODER_VOCAB,
    pe_input=1000,
    pe_target=1000,
    rate=dropout_rate,
)

# Define checkpoint path
checkpoint_path = "summary_datas/checkpoints"
# Define the checkpoint object
ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
# Define checkpoint manager
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
# Check if latest checkpoint exists
if ckpt_manager.latest_checkpoint:
    # Restore the latest checkpoint
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print("Latest checkpoint restored!!")
else:
    print("No checkpoint found. Initializing from scratch.")


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask


def create_masks(inp, tar):
    enc_padding_mask = create_padding_mask(inp)
    dec_padding_mask = create_padding_mask(inp)

    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask


# def summarize(input_article):
#     input_article = preprocess(input_article)

#     # Tokenize input article
#     input_article_sequence = article_tokenizer.texts_to_sequences([input_article])
#     input_article_sequence = tf.keras.preprocessing.sequence.pad_sequences(input_article_sequence,
#                                                                             maxlen=ENCODER_LEN,
#                                                                             padding='post', truncating='post')

#     encoder_input = tf.expand_dims(input_article_sequence[0], 0)

#     # Initialize decoder input
#     decoder_input = [summary_tokenizer.word_index['<sos>']]
#     output = tf.expand_dims(decoder_input, 0)

#     for i in range(DECODER_LEN):
#         enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input, output)

#         # Predict using the loaded model
#         predictions, attention_weights = transformer(
#             encoder_input,
#             output,
#             False,
#             enc_padding_mask,
#             combined_mask,
#             dec_padding_mask
#         )

#         predictions = predictions[: ,-1:, :]
#         predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

#         if predicted_id == summary_tokenizer.word_index['<eos>']:
#             return tf.squeeze(output, axis=0), attention_weights

#         output = tf.concat([output, predicted_id], axis=-1)

#     summarized = tf.squeeze(output, axis=0).numpy()
#     summarized = np.expand_dims(summarized[1:], 0)

#     return summary_tokenizer.sequences_to_texts(summarized)[0]


def evaluate(input_article):
    input_article = article_tokenizer.texts_to_sequences([input_article])
    input_article = tf.keras.preprocessing.sequence.pad_sequences(
        input_article, maxlen=ENCODER_LEN, padding="post", truncating="post"
    )

    encoder_input = tf.expand_dims(input_article[0], 0)

    decoder_input = [summary_tokenizer.word_index["<sos>"]]
    output = tf.expand_dims(decoder_input, 0)

    for i in range(DECODER_LEN):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
            encoder_input, output
        )

        predictions, attention_weights = transformer(
            encoder_input,
            output,
            False,
            enc_padding_mask,
            combined_mask,
            dec_padding_mask,
        )

        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        if predicted_id == summary_tokenizer.word_index["<eos>"]:
            return tf.squeeze(output, axis=0), attention_weights

        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0), attention_weights


def summarize(input_article):
    input_article = preprocess(input_article)
    summarized = evaluate(input_article=input_article)[0].numpy()
    summarized = np.expand_dims(summarized[1:], 0)
    return summary_tokenizer.sequences_to_texts(summarized)[0]


# title = "4 ex-bank officials booked for cheating bank of ₹209 crore "
# print(summarize(title))
