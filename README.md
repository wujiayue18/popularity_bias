# Popularity-Steer for Mitigating Popularity Bias

This repository is adapted from the official [LightGCN-PyTorch implementation](https://github.com/gusye1234/LightGCN-PyTorch), with modifications to address popularity bias in recommendation systems.

---

## 🔧 Modifications

* **Popularity-Steer Mechanism**: Extracts popularity-related signals to disentangle them from item embeddings.
* **Bias Mitigation at Inference**: Refines item embeddings during inference to reduce popularity bias while preserving overall performance.
* **Integration Enhancements**: Adjustments to ensure smooth integration of the new mechanism into the LightGCN backbone.

---

## 🚀 Model Training

### 🔹 LightGCN Baseline

Pre-trained LightGCN models are stored in: "./checkpoints/{dataset}/lgn-amazon-book-3-64-lgn.pth.tar"

### Popularity-Steer Adapter

This repository implements a **Popularity-Steer Adapter** inspired by the adapter design in:

> Han, Chi, et al. *LM-Switch: Lightweight Language Model Conditioning in Word Embedding Space.* arXiv preprint arXiv:2305.12798, 2023.


* **Gowalla**

```bash
python main.py --dataset gowalla --epsilon 1 --alpha 10 --beta 1 --gamma 100 --epochs 50
```

* **Yelp2018**

```bash
python main.py --dataset yelp2018 --epsilon 2 --alpha 20 --beta 1 --gamma 100 --epochs 60
```

* **Amazon-Book**

```bash
python main.py --dataset amazon-book --epsilon 2 --alpha 10 --beta 1 --gamma 100 --epochs 100
```

---

## ✅ Model Testing

## test lightgcn
```bash
python main.py --load 1 --steer 0 --dataset gowalla 
```

## test steer

```bash
python main.py --load 1 --dataset gowalla --epsilon 1 --alpha 10 --beta 1 --gamma 100
python main.py --load 1 --dataset yelp2018 --epsilon 2 --alpha 20 --beta 1 --gamma 100
python main.py --load 1 --dataset amazon-book --epsilon 2 --alpha 10 --beta 1 --gamma 100
```



## 📂 Code Base

This work builds upon the original **LightGCN** repository: [LightGCN-PyTorch](https://github.com/gusye1234/LightGCN-PyTorch).

