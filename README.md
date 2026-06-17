# Hybrid-CNN-Swin-Skin-Cancer
 This repository hosts the final audited implementation of our 6-Pillar Heterogeneous Fusion Engine for highly robust skin lesion and melanoma classification. By unifying latent feature representations from four state-of-the-art vision backbones, a simulated multi-scale UNet skip architecture, and processed clinical patient metadata, the network maps an optimized, multi-modal unified vector space of $5760$ dimensions.Equipped with Asymmetric Focal Loss, Weight EMA, and Monte Carlo (MC) Dropout soft-label generation, this architecture resolves complex feature alignment issues and achieves an 82%+ Macro F1-score across highly unbalanced diagnostic distributions. google drive link for the files:-https://drive.google.com/drive/folders/1CPhVgC9VAVWdPyAMvyVQTqnWZMC49xi3?usp=drive_link


model pipeline:-
```text   INPUT: CLINICAL IMAGE & PATIENT METADATA
                                           │
         ┌─────────────────────────────────┴─────────────────────────────────┐
         ▼                                                                   ▼
 ┌──────────────┐                                                    ┌──────────────┐
 │ VISION PATH  │                                                    │METADATA PATH │
 └──────┬───────┘                                                    └──────┬───────┘
        │                                                                   │
        ├──► Swin Transformer ──────► Latent Vector (768)  ──► LayerNorm    ├──► Localization (15 classes)
        ├──► ConvNeXt-Base   ──────► Latent Vector (1024) ──► LayerNorm    │     └──► Embedding (16)
        ├──► EfficientNetV2  ──────► Latent Vector (1792) ──► LayerNorm    │
        ├──► DenseNet        ──────► Latent Vector (1920) ──► LayerNorm    ├──► Sex (3 classes)
        │                                                                   │     └──► Embedding (8)
        └──► UNet (ResNet34) ──────► Latent Vector (448)                   │
              └──► LayerNorm ──► Linear(448→128) ──► Tanh                  └──► Age (Continuous)
                    └──► Dropout(0.3) ──► [128 Dims]                       │     └──► Z-Score Normalization (1)
                                │                                           │
                                ▼                                           ▼
                       [5504 Vision Dims]                        ┌─────────────────────┐
                                │                                │  Metadata MLP Block │
                                │                                │ ─────────────────── │
                                │                                │  Linear(25 → 128)   │
                                │                                │  GELU Activation    │
                                │                                │  LayerNorm Layer    │
                                │                                └──────────┬──────────┘
                                │                                           │
                                │                                           ▼
                                │                                   [128 Meta Dims]
                                │                                           │
                                └───────────────────┬───────────────────────┘
                                                    │
                                                    ▼
                                       ┌─────────────────────────┐
                                       │  UNIFIED VECTOR SPACE   │
                                       │ ─────────────────────── │
                                       │ Concat: 5504 + 128 Dims │
                                       │   Total: 5760 Tensors   │
                                       └────────────┬────────────┘
                                                    │
                                                    ▼
                                       ┌─────────────────────────┐
                                       │ GRANDMASTER FUSION HEAD │
                                       │ ─────────────────────── │
                                       │   Linear(5760 → 1024)   │
                                       │   LayerNorm + GELU      │
                                       │   Dropout Layer (0.4)   │
                                       │   Linear(1024 → 256)    │
                                       │   LayerNorm + GELU      │
                                       │   Dropout Layer (0.3)   │
                                       │   Linear(256 → 7)       │
                                       └────────────┬────────────┘
                                                    │
                                                    ▼
                                       ┌─────────────────────────┐
                                       │     SOFTMAX OUTPUT      │
                                       │  7 Skin Cancer Classes  │
                                       └─────────────────────────┘
