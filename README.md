# Transformer from Scratch

Built a full transformer (Attention Is All You Need) from scratch in PyTorch.

## Architecture
- Scaled dot-product attention
- Multi-head attention  
- Positional encoding
- Encoder/Decoder blocks
- 68M parameters

## Training
- Dataset: Multi30k (EN → DE)
- Epochs: 10
- Final val loss: 0.42
- Hardware: Kaggle T4 GPU
- Time per epoch: ~90 seconds

## Results
Model learns to translate but repeats tokens — 
beam search decoding to be added next.

## Next steps
- Beam search decoding
- Train on larger dataset
- Compare with LSTM baseline
