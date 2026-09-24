# Chatbot: Seq2Seq Architecture Comparison

A chatbot built from scratch on the Cornell Movie-Dialogs Corpus, comparing three architectures — LSTM seq2seq, GRU with attention, and (planned) a fine-tuned Transformer — to see firsthand how each generation of sequence modeling handles conversational response generation.

## Problem

Given a conversational input, generate a plausible response. Rather than jumping straight to a pretrained Transformer, this project deliberately builds the "before" architectures first — plain LSTM seq2seq, then attention — to see the real, hands-on limitations each one has, not just read about them.

## Dataset

Cornell Movie-Dialogs Corpus: ~220,000 conversational exchanges extracted from movie scripts.

**Data pipeline:**

- Parsed `movie_lines.txt` and `movie_conversations.txt` (note: `iso-8859-1` encoding, not UTF-8) into a line lookup and conversation structure
- Built 221,282 (input, response) pairs from consecutive dialogue turns
- Length analysis showed a median of 7 words but a long tail up to 300+ words; filtered to pairs ≤32 words on both sides, keeping 200,050 pairs
- Vocabulary analysis found 138,538 unique tokens; cut to 26,625 words appearing more than 5 times (36% of all tokens were singletons — typos, rare names, one-off phrasing), plus `<PAD>`, `<START>`, `<END>`, `<UNK>` special tokens
- Sequences padded to a fixed length of 33 tokens

## Models

### 1. LSTM Seq2Seq (baseline, no attention)

Standard encoder-decoder: the encoder compresses the entire input into a single fixed-size vector, and the decoder generates the response from that vector alone.

- 21.5M parameters, trained with EarlyStopping and ReduceLROnPlateau
- Validation accuracy: 78.06% _(inflated by padding-token dominance in the sequences — not a reliable measure of true response quality on its own)_
- **Observed behavior:** generic, repetitive responses regardless of input (e.g. "I don't know." to multiple unrelated questions) — a textbook symptom of the single fixed-vector bottleneck, since the decoder has no way to look back at specific input words.

### 2. GRU + Attention

Same overall structure, but the encoder now exposes its full sequence of outputs, and the decoder computes attention over them at every generation step — intended to let the model focus on relevant input words dynamically, rather than relying on one compressed summary.

- 28M parameters, same training setup as the LSTM baseline for a fair comparison
- Validation accuracy: comparable to the baseline
- **Observed behavior:** the same generic/repetitive response pattern as the LSTM baseline, tested on identical inputs. Attention did not visibly resolve the issue in this run — an open question we're still investigating (possible causes under consideration: decoder state not carrying enough information across generation steps at inference time, or the model needing more training/data to make full use of attention).

### 3. Transformer (fine-tuned) — in progress

Planned: fine-tune a pretrained conversational/causal language model (DialoGPT, then DistilGPT2) on the same data. Currently blocked by a training instability bug (validation loss diverging to NaN), traced to how the tokenizer's padding token interacts with the loss-masking logic during fine-tuning. Under investigation.

## Honest Comparison Notes

Per-token validation accuracy is a weak proxy for real conversational quality here — a model can score well by learning to predict `<PAD>` tokens correctly (which dominate most sequences after padding to length 33) while still producing poor actual responses. The qualitative test — reading real generated responses to fixed test inputs — was far more revealing than the accuracy number for either model.

## Tech Stack

- **Deep Learning**: TensorFlow / Keras
- **NLP preprocessing**: NLTK, custom tokenization/vocabulary pipeline
- **Deployment**: Streamlit
- **Training environment**: local (EDA/preprocessing) + Google Colab (GPU-dependent training, since local Python 3.14 has no TensorFlow wheel yet)

## Project Structure

```
chatbot/
├── data/
│   ├── raw/                       # Cornell corpus files
│   └── processed/                 # padded sequences, vocab (.npy/.pkl)
├── notebooks/
│   ├── 01_02_data_collection_and_preprocessing.ipynb
│   ├── 03_lstm_seq2seq.ipynb
│   ├── 04_gru_attention.ipynb
│   └── 05_transformer_finetuned.ipynb   # in progress
├── models/
│   ├── lstm_seq2seq/
│   └── gru_attention/
├── app.py                          # Streamlit comparison app
├── requirements.txt
└── README.md
```

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Key Learnings

- Token-level training metrics can be misleading for sequence generation tasks — always sanity-check with real, readable output, not just the loss/accuracy curve.
- The "fixed vector bottleneck" that motivated attention mechanisms is visible in practice, not just theoretical — the baseline model's generic responses were a direct, observable symptom.
- Adding attention doesn't automatically fix generation quality — architecture alone isn't sufficient; getting real improvement likely also depends on training duration, data scale, and correct inference-time state handling, which is exactly what we're still debugging.
- Fine-tuning a pretrained causal LM has its own failure modes distinct from training from scratch (e.g. tokenizer pad/eos token conflicts) — worth understanding rather than treating `Trainer.train()` as a black box.

## What's Next

- Resolve the Transformer fine-tuning training instability and complete the 3-way comparison
- Investigate why attention didn't visibly improve the GRU model's output quality
- Try beam search or top-k/top-p sampling instead of greedy argmax decoding, which may reduce repetitive "safe" responses
- Follow up with a code-generation project applying the same architecture comparison to a harder task
