# ai-learning-repo
This repo contains code related to my learnings in GenAI. I've following these tasks and I will be updating each task in a python file

## Tasks
Code. Plot. Break. Repeat.  

---

## Tokenization & Embeddings
- [x] Build a byte-pair encoder (BPE) and train your own subword vocab  
- [ ] Write a token visualizer to map words/chunks → token IDs  
- [ ] Compare one-hot vs learned embeddings, and  
  - [ ] Plot cosine distances between token vectors  

---

## Positional Embeddings
- [ ] Implement & compare:
  - [ ] Sinusoidal  
  - [ ] Learned  
  - [ ] RoPE  
  - [ ] ALiBi  
- [ ] Animate a toy sequence being position-encoded in 3D  
- [ ] Ablate positions — watch attention collapse  

---

## Self-Attention & Multihead Attention
- [ ] Hand-wire dot-product attention for one token  
- [ ] Scale to multi-head attention, plot per-head weight heatmaps  
- [ ] Mask out future tokens, verify causal property  

---

## Transformers: Q, K, V & Stacking
- [ ] Stack your Attention + LayerNorm + Residual → single-block transformer  
- [ ] Generalize to n-block “mini-former” on toy data  
- [ ] Dissect Q, K, V — swap them, break them, see what explodes  

---

## Sampling Parameters: Temperature / Top-k / Top-p
- [ ] Build a sampler dashboard — interactively tune temp/k/p  
- [ ] Plot entropy vs output diversity  
- [ ] Set temp = 0 (argmax) — watch repetition set in  

---

## KV Cache (Fast Inference)
- [ ] Record & reuse KV states; measure speedup vs no-cache  
- [ ] Build a cache hit/miss visualizer for token streams  
- [ ] Profile cache memory cost for long vs short sequences  

---

## Long-Context Tricks: Infini-Attention / Sliding Window
- [ ] Implement sliding-window attention; measure loss on long docs  
- [ ] Benchmark memory-efficient attention (recompute, flash)  
- [ ] Plot perplexity vs context length, find the context collapse point  

---

## Mixture of Experts (MoE)
- [ ] Code a 2-expert router layer; route tokens dynamically  
- [ ] Plot expert utilization histograms  
- [ ] Simulate sparse vs dense routing, measure FLOP savings  

---

## Grouped Query Attention
- [ ] Convert your mini-former to grouped query layout  
- [ ] Measure speed vs vanilla multi-head on large batch  
- [ ] Ablate number of groups, plot latency  

---

## Normalization & Activations
- [ ] Hand-implement:
  - [ ] LayerNorm  
  - [ ] RMSNorm  
  - [ ] SwiGLU  
  - [ ] GELU  
- [ ] Ablate each — observe train/test loss impact  
- [ ] Plot activation distributions layer-wise  

---

## Pretraining Objectives
- [ ] Train:
  - [ ] Masked LM  
  - [ ] Causal LM  
  - [ ] Prefix LM  
- [ ] Plot loss curves — compare who learns “English” faster  
- [ ] Generate samples from each — note quirks  

---

## Finetuning vs Instruction Tuning vs RLHF
- [ ] Fine-tune on a small custom dataset  
- [ ] Instruction-tune by prepending tasks (e.g., “Summarize: …”)  
- [ ] RLHF:
  - [ ] Hack a reward model  
  - [ ] Run PPO for 10 steps  
  - [ ] Plot reward curve  

---

## Scaling Laws & Model Capacity
- [ ] Train tiny / small / medium models  
- [ ] Plot loss vs size  
- [ ] Benchmark time, VRAM, throughput  
- [ ] Extrapolate scaling curve — how “dumb” can you go?  

---

## Quantization
- [ ] Implement PTQ (Post-Training Quantization)  
- [ ] Implement QAT (Quantization-Aware Training)  
- [ ] Export to GGUF / AWQ  
- [ ] Plot accuracy drop vs compression ratio  

---

## Inference / Training Stacks
- [ ] Port a model from HuggingFace → Deepspeed → vLLM → ExLlama  
- [ ] Profile throughput, VRAM, latency across all three  

---

## Synthetic Data
- [ ] Generate toy data, add noise, dedupe, create eval splits  
- [ ] Visualize model learning curves on real vs synthetic data  

---
