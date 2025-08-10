# Attention-Based Neural Machine Translation: English to Pig Latin

A PyTorch implementation of a transformer decoder with attention mechanisms for character-level sequence-to-sequence translation from English to Pig Latin.

## 🎯 Project Overview

This project implements a neural machine translation model that learns the rules of Pig Latin transformation implicitly from English-Pig Latin word pairs. The model demonstrates the power of attention mechanisms in handling character-level sequence transformations.

### 🐷 What is Pig Latin?

Pig Latin is a simple English transformation with these rules:
- **Consonant start**: Move consonant(s) to end + "ay" → `team` → `eamtay`
- **Vowel start**: Add "way" to end → `impress` → `impressway`  
- **Consonant clusters**: Move entire clusters → `shopping` → `oppingshay`

## 🏗️ Architecture

### Core Components

1. **Scaled Dot-Product Attention**
   - Computes attention weights using query-key similarity
   - Implements the fundamental attention mechanism: `Attention(Q,K,V) = softmax(QK^T/√d)V`

2. **Causal Scaled Dot-Product Attention** 
   - Masks future tokens to prevent information leakage
   - Essential for autoregressive decoding

3. **Transformer Decoder**
   - Multi-layer architecture with residual connections
   - Self-attention followed by encoder-decoder attention
   - ReLU feedforward layers

### Model Pipeline
```
Input → GRU Encoder → Transformer Decoder → Output Prediction
              ↓
        Attention Visualization
```

## 📊 Dataset

- **Source**: 6,387 unique word pairs from Jane Austen's "Sense and Sensibility"
- **Vocabulary**: 29 tokens (26 letters + dash + `<SOS>` + `<EOS>`)
- **Examples**: `{(the, ethay), (family, amilyfay), (of, ofway)}`
- **Batching**: Grouped by sequence length for efficient processing

## 🚀 Key Features

### Advanced Attention Mechanisms
- **Multi-head attention** with query, key, value projections
- **Residual connections** for gradient flow optimization
- **Causal masking** for proper autoregressive generation

### Attention Visualization
- **Heatmap generation** showing model focus during translation
- **Success/failure analysis** across different word categories:
  - Single consonants (`cake`)
  - Consonant clusters (`drink`) 
  - Compound words (`well-mannered`)
  - Unusual combinations (`aardvark`)

### Model Interpretability
- Character-level attention weight visualization
- Analysis of learning patterns and failure modes
- Insight into transformer behavior on linguistic transformations

## 📈 Performance

### Training Results
- **Training time**: ~2 minutes on AMD Ryzen 7 5800U
- **Architecture**: 3-layer transformer decoder with 64 batch size, 20 hidden dimensions
- **Convergence**: Training loss dropped from 2.0+ to ~0.003, validation loss to ~0.15
- **Final performance**: Strong convergence with minimal overfitting

### Model Behavior Analysis
**Successful Translations:**
- Simple words: `street`, `bake`, `drive` → Perfect Pig Latin conversion
- Compound patterns: `dogcat` (made-up word) → Correctly applies rules to novel combinations
- Complex sentences: `"nix on the real name"` → `"ixnay onway ethay ealray amenay"`

**Failure Cases:**
- **Rare combinations**: `aardvark` → Struggles with unusual `aard` pattern
- **Long compound words**: `ill-mannered` → Nearly correct but drops one `n`
- **Unusual consonants**: `xylophone` → `xy` cluster outside training vocabulary
- **Repetitive patterns**: `drrrrive` → Cannot handle 4 consecutive `r`s properly

### Attention Pattern Insights
- **Character alignment**: Clear diagonal patterns show proper character-to-character mapping
- **Copy mechanism**: Model successfully learns to copy and reorder characters
- **Failure modes**: Attention becomes diffuse for rare letter combinations and long sequences

## 🛠️ Implementation Details

### Core Classes
```python
class ScaledDotAttention(nn.Module)
class CausalScaledDotAttention(nn.Module)  
class TransformerDecoder(nn.Module)
```

### Key Technical Features
- **Batch matrix multiplication** (`torch.bmm`) for efficient attention computation
- **Upper triangular masking** (`torch.triu`) for causal attention
- **Parameter-efficient design** suitable for educational and research purposes

## 📋 Usage

```python
# Train the model
python nmt.py --train=True

# Visualize attention (load pre-trained model)
python nmt.py --train=False --visualize=True

# Test custom words for analysis
TEST_WORD_ATTN = ["street", "bake", "aardvark", "dogcat"]
```

### Reproducing Results
The training shows consistent patterns across runs:
- **Epoch 1-10**: Rapid loss decrease and basic pattern learning
- **Epoch 10-50**: Refinement of character-level mappings  
- **Epoch 50-100**: Fine-tuning with occasional overfitting signs
- **Final result**: Successful translation of common patterns, struggles with edge cases

## 🔍 Attention Analysis

The project includes comprehensive attention visualization demonstrating model interpretability:

### Success Patterns
- **Clear diagonal attention**: For simple words like `bake` and `drive`, attention weights form clean diagonal patterns showing direct character correspondence
- **Proper rule learning**: Model correctly identifies consonant vs. vowel patterns and applies appropriate Pig Latin transformations
- **Generalization**: Successfully translates made-up words like `dogcat` by combining learned patterns from `dog` and `cat`

### Failure Mode Analysis
- **Rare letter combinations**: `aardvark` shows scattered attention due to unusual `aard` sequence not well-represented in training
- **Length limitations**: `ill-mannered` translation is mostly correct but drops characters in longer sequences
- **Repetitive patterns**: `drrrrive` with 4 consecutive `r`s confuses the attention mechanism
- **Out-of-vocabulary**: `xylophone` with `xy` cluster falls outside training distribution

### Attention Heatmaps
The visualization reveals:
- **Character-to-character mapping** during translation process
- **Attention weight distributions** across input sequences  
- **Model confidence patterns** through attention intensity
- **Error correlation** with attention diffusion in failure cases

### Key Findings
- Strong attention alignment correlates with successful translations
- Attention becomes diffuse when encountering rare patterns
- Model learns implicit Pig Latin rules through character-level attention
- Failure cases provide insights into model limitations and training data gaps

## 🎓 Educational Value

This implementation serves as an excellent introduction to:
- **Transformer architectures** and attention mechanisms
- **Sequence-to-sequence modeling** at character level
- **Neural machine translation** fundamentals
- **Model interpretability** through attention visualization
- **PyTorch** deep learning implementation patterns

## 📄 Academic Context

Developed as part of CS 72/LING 48 Accelerated Computational Linguistics at Dartmouth College (Winter 2024). Based on foundational work by Jimmy Ba, Roger Grosse, and Paul Vicol.

## 🔧 Requirements

- PyTorch
- NumPy
- Matplotlib (for visualizations)
- Python 3.7+

## 📚 Key Learnings

- **Attention mechanisms** enable models to focus on relevant input tokens
- **Character-level modeling** is effective for morphological transformations
- **Residual connections** are crucial for training deeper networks
- **Visualization tools** provide valuable insights into model behavior
- **Causal masking** is essential for proper autoregressive generation

---

*This project demonstrates the power of attention-based neural networks in learning linguistic patterns and provides hands-on experience with modern NLP architectures.*
