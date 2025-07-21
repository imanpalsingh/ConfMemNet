# ConfMemNet

**Attention is Not Enough: Confidence-Guided Dual Memory for Context-Aware Transformers**

ConfMemNet is a lightweight Transformer architecture that integrates a novel confidence-based gating mechanism and a dual memory module to improve contextual relevance and long-range dependency tracking. This project reproduces the results from our accompanying paper and provides an implementation of the model.


## Motivation

While traditional Transformer-based models like GPT excel at attention, they lack a principled mechanism to decide what to retain. ConfMemNet introduces **confidence-aware token evaluation** and dual memory buffers (salient and long-term) to store and reuse meaningful context, mimicking aspects of human cognition.



## Installation

```bash
git clone https://github.com/imanpalsingh/ConfMemNet.git
cd ConfMemNet
pip install -r requirements.txt
```

Ensure nltk downloads stopword corpus:
```
import nltk
nltk.download("stopwords")
```

## Dataset
This project uses the AG News dataset via the datasets library. The training subset is limited to 6,000 samples (5,000 train, 1,000 validation) for faster reproducibility.

## Training
To run training:

```bash
python main.py
```
All configuration values are defined in confmemnet/config.py. These include:

```python
{
  "vocab_size": 50257,
  "max_seq_len": 32,
  "embedding_dim": 64,
  "ffn_dim": 128,
  "num_layers": 1,
  "salient_threshold": 0.6,
  "long_term_threshold": 0.85,
  "batch_size": 16,
  "learning_rate": 2e-4,
  "num_epochs": 20,
  "stopword_penalty": 0.1,
  "freq_penalty": 0.6
}
```

## Results
Validation loss curve:
|Epoch|LSTM Loss|ConfMemNet Loss|
|---|---|---|
1|7.9370|7.8073|
10|7.2608|3.2411|
20|6.9618|1.4622|

See full paper for training curves and interpretation.

## Architecture Highlights
1. Confidence Gate: Learns to evaluate token relevance
2. Dual Memory: Salient and long-term memory buffers
3. Attention Routing: Soft interpolation of memory sources
4. Stopword & Frequency Penalization: Regularizes overconfident common tokens

## Citation
If you use this work in your research, please cite the accompanying paper:

```bib
@misc{2025confmemnet,
  title   = {Attention is Not Enough: Confidence-Guided Dual Memory for Context-Aware Transformers},
  author  = {Imanpal Singh},
  year    = {2025},
  url     = {https://github.com/imanpalsingh/ConfMemNet}
}
```

## Contributions
Open to improvements on scaling, multi-head implementation, and curriculum learning!

## License
MIT License