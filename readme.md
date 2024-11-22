## Contrastive Unlearning with MOCO


This repository implements a contrastive unlearning approach using MOCO (Momentum Contrast) framework. The main implementation is in `unlearn.py`, which contains:

- `MOCO_Unlearn` class that handles the contrastive unlearning process:
  - Maintains a queue of feature embeddings and corresponding labels
  - Implements contrastive loss between unlearning samples and queue samples
  - Combines contrastive loss with cross-entropy loss for retained samples

The `core` library is shared with the contrastive unlearning implementation.