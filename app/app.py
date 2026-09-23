"""
Streamlit app for the Chatbot project.
Lets the user chat with both the LSTM seq2seq baseline and the
GRU + Attention model side by side for direct comparison.
"""

import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Attention, Concatenate
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LEN = 33
UNITS = 256

st.set_page_config(page_title="Chatbot Comparison", page_icon="💬", layout="wide")


# ---------------------------------------------------------------------
# Cached loading — runs once per session, not on every interaction
# ---------------------------------------------------------------------

@st.cache_resource
def load_vocab():
    word2idx = joblib.load("data/processed/word2idx.pkl")
    idx2word = joblib.load("data/processed/idx2word.pkl")
    return word2idx, idx2word


def sentence_to_ids(sentence, word2idx):
    return [word2idx.get(word, word2idx["<UNK>"]) for word in sentence.lower().split()]


@st.cache_resource
def load_gru_attention_models():
    """Loads the trained GRU+Attention model and rebuilds inference sub-models."""
    full_model = load_model("models/gru_attention.h5")

    encoder_inputs = full_model.input[0]
    encoder_gru = full_model.get_layer("encoder_gru")
    encoder_outputs, state_h = encoder_gru.output
    encoder_model = Model(encoder_inputs, [encoder_outputs, state_h])

    decoder_inputs = full_model.input[1]
    decoder_state_input_h = Input(shape=(UNITS,))
    encoder_outputs_input = Input(shape=(None, UNITS))

    decoder_embedding_layer = full_model.get_layer("decoder_embedding")
    decoder_gru = full_model.get_layer("decoder_gru")
    decoder_dense = full_model.get_layer("decoder_dense")

    dec_emb = decoder_embedding_layer(decoder_inputs)
    dec_gru_out, dec_state_h = decoder_gru(dec_emb, initial_state=decoder_state_input_h)
    attn_out = Attention()([dec_gru_out, encoder_outputs_input])
    concat_out = Concatenate(axis=-1)([dec_gru_out, attn_out])
    dec_outputs = decoder_dense(concat_out)

    decoder_model = Model(
        [decoder_inputs, encoder_outputs_input, decoder_state_input_h],
        [dec_outputs, dec_state_h],
    )

    return encoder_model, decoder_model


# ---------------------------------------------------------------------
# Generation functions
# ---------------------------------------------------------------------


def generate_gru_attention_response(input_text, encoder_model, decoder_model, word2idx, idx2word, max_len=MAX_LEN):
    input_seq = pad_sequences([sentence_to_ids(input_text, word2idx)], maxlen=max_len, padding="post")
    enc_outs, state_h = encoder_model.predict(input_seq, verbose=0)

    target_seq = np.array([[word2idx["<START>"]]])
    generated = []

    for _ in range(max_len):
        output_tokens, h = decoder_model.predict([target_seq, enc_outs, state_h], verbose=0)
        sampled_idx = np.argmax(output_tokens[0, -1, :])
        sampled_word = idx2word.get(sampled_idx, "<UNK>")

        if sampled_word == "<END>":
            break

        generated.append(sampled_word)
        target_seq = np.array([[sampled_idx]])
        state_h = h

    return " ".join(generated) if generated else "(no response generated)"


# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------

st.title("💬 Chatbot Architecture Comparison")
st.caption(
    "Trained from scratch on the Cornell Movie-Dialogs Corpus. "
)

with st.expander("ℹ️ About this project"):
    st.markdown(
        """
        This app compares two chatbot architectures trained on the same data:

        - **LSTM Seq2Seq (baseline)** — encoder compresses the input into a single
          fixed-size vector; the decoder generates a response from that alone.
        - **GRU + Attention** — the decoder can look back at every input word at
          each generation step, rather than relying on one fixed summary.

        **A known limitation of both models right now:** responses can be generic
        or repetitive (e.g. "I don't know."). This is partly expected for
        small-scale models trained from scratch on a modest dataset, and partly
        an open issue — attention did not visibly outperform the baseline yet in
        our testing, which we're still investigating. A third, Transformer-based
        model (fine-tuned) is planned but currently blocked by a training bug.
        """
    )

user_input = st.text_input("Type something to say to the bots:", placeholder="how are you")

col1, col2 = st.columns(2)

if st.button("Send", type="primary") and user_input.strip():
    word2idx, idx2word = load_vocab()

    with col1:
        st.subheader("🔸 GRU + Attention")
        with st.spinner("Generating..."):
            gru_encoder, gru_decoder = load_gru_attention_models()
            response = generate_gru_attention_response(user_input, gru_encoder, gru_decoder, word2idx, idx2word)
        st.write(response)

st.divider()
st.caption("Built with TensorFlow/Keras · Trained on the Cornell Movie-Dialogs Corpus")