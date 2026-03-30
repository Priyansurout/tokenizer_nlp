# NLP Tokenizer — Tokenization Gap

## 📌 Assignment Details
- **Subject:** NLP  
- **Submitted by:** Priyansu rout
- **Roll No.:** 23052012  
- **Section:** CSE-39  

---

## Project Description
This project implements and compares **five different tokenization strategies** as part of the NLP assignment on the *Tokenization Gap*. The goal is to evaluate how different tokenizers handle vocabulary size, out-of-vocabulary (OOV) words, compression, and runtime performance on the same corpus.

---
## Tokenizers Implemented

| Tokenizer | Strategy |
|-----------|----------|
| **WordPiece** *(baseline)* | Greedy longest-match from left; splits rare words into `##`-prefixed subwords |
| **BPE (Byte-Pair Encoding)** | Iteratively merges the most frequent adjacent character pairs |
| **Character-level** | Splits every word into individual characters; zero OOV by design |
| **Hybrid (Word + Char)** | Uses whole-word tokens for frequent words; falls back to character-level for rare/OOV words |
| **Dynamic Merging** | BPE with online domain adaptation — learns extra merge rules from a new corpus |

---

## ⚙️ Features

- ✅ Five tokenizer implementations with a shared `BaseTokenizer` interface
- ✅ Unified `encode()` and `decode()` methods across all tokenizers
- ✅ Automatic performance benchmarking via `TokenizerEvaluator`
- ✅ Metrics tracked: vocab size, avg tokens/sentence, compression ratio, OOV rate, train & inference time
- ✅ Domain adaptation support (Dynamic Merging tokenizer)
- ✅ Side-by-side tokenization output for easy comparison

---

## 🚀 How to Run

1. **Clone the repository:**
```bash
   git clone https://github.com/Aryan-20-04/NLP-tokenizer.git
   cd NLP-tokenizer
```

2. **Install dependencies** *(standard library only — no pip installs needed):*
```bash
   python --version   
```

3. **Run the demo:**
```bash
   python tokenizer.py
```


