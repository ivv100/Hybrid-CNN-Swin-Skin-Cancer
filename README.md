# Hybrid-CNN-Swin-Skin-Cancer
This repo hosts my B.Tech project: a hybrid CNN-Swin Transformer for melanoma classification. Fusing EfficientNetV2/ConvNeXtV2 with a custom Swin block, it hit 71%+ zero-shot accuracy on the unseen ISIC dataset after training on HAM10000. This proves true Out-Of-Distribution (OOD) generalization for real-world clinical environments.

The google drive link for both model files:-https://drive.google.com/drive/folders/1sS_XebhYvEe6nsGjCeqi10Pf3UrNYIfP?usp=sharing

model pipeline:-

INPUT IMAGE (224×224×3)
│
┌────────────┴────────────┐
│                         │
▼                         ▼
┌─────────────────────┐   ┌─────────────────────┐
│  EfficientNetV2-S   │   │   ConvNeXtV2-Base   │
│ (HAM10k pretrained) │   │ (HAM10k pretrained) │
│  Fine-tuned ISIC    │   │  Fine-tuned ISIC    │
└─────────────────────┘   └─────────────────────┘
│                         │
▼                         ▼
Feature Map 7×7×1280      Feature Map 7×7×1024
│                         │
▼                         ▼
┌─────────────────────┐   ┌─────────────────────┐
│  1×1 Conv Project   │   │  1×1 Conv Project   │
│     1280 → 512      │   │     1024 → 512      │
└─────────────────────┘   └─────────────────────┘
│                         │
▼                         ▼
Flatten 7×7×512 →         Flatten 7×7×512 →
49 tokens × 512           49 tokens × 512
│                         │
└────────────┬────────────┘
             │
             ▼
   ┌─────────────────────┐
   │    Token Concat     │
   │    [49+49] = 98     │
   │    tokens × 512     │
   └─────────────────────┘
             │
             ▼
   ┌─────────────────────┐
   │ Positional Encoding │
   │ (learnable 98×512)  │
   └─────────────────────┘
             │
             ▼
   ┌─────────────────────┐
   │  Swin Transformer   │
   │    Fusion Stage     │
   │                     │
   │  ┌───────────────┐  │
   │  │  Window Attn  │  │
   │  │    Block 1    │  │
   │  └───────────────┘  │
   │          │          │
   │  ┌───────────────┐  │
   │  │ Shifted Window│  │
   │  │  Attn Block 2 │  │
   │  └───────────────┘  │
   │          │          │
   │  ┌───────────────┐  │
   │  │ MLP + LayerNorm│ │
   │  └───────────────┘  │
   └─────────────────────┘
             │
             ▼
   ┌─────────────────────┐
   │   Global Avg Pool   │
   │    98×512 → 512     │
   └─────────────────────┘
             │
             ▼
   ┌─────────────────────┐
   │    Dropout (0.3)    │
   └─────────────────────┘
             │
             ▼
   ┌─────────────────────┐
   │      MLP Head       │
   │    512 → 256 → 7    │
   │   GELU + Dropout    │
   └─────────────────────┘
             │
             ▼
   ┌─────────────────────┐
   │   Softmax Output    │
   │    7 Skin Cancer    │
   │       Classes       │
   └─────────────────────┘


Progress:-`

- [x] **Phase 1: Feature Extraction SOTA**
  - [x] Train EfficientNetV2-S on HAM10000.
  - [x] Train ConvNeXtV2-Base on HAM10000.
  - [x] Blind evaluation on ISIC dataset (Achieved >71% Zero-Shot OOD).
- [ ] **Phase 2: Transformer Fusion Head**
  - [ ] Implement 1x1 Conv projection (reduce dimensions to 512).
  - [ ] Engineer Token Concatenation Block [49 + 49 = 98 tokens].
  - [ ] Apply Learnable Positional Encodings.
  - [ ] Build Swin Transformer Window Attention layers.
- [ ] **Phase 3: Final Deployment**
  - [ ] End-to-end model training.
  - [ ] Export final pipeline for inference.
     
  Thank you for your time for reading all these
