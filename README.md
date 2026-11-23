# ESRepNet: An Efficient Scaled Deep Learning Model for Tiny Agricultural Pest Detection

[![Paper](https://img.shields.io/badge/ paper-Applied%20Intelligence%202026-blue)](https://doi.org/10.1007/s10489-026-07559-6)
[![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/suzauddola/ESRepNet)

Official implementation repository accompanying the paper:

> **An efficient scaled deep learning model for detection of pests in complex agricultural environment**  
> Md Suzauddola, Junde Chen, Adnan Zeb, and Defu Zhang  
> *Applied Intelligence*, 56, Article 562 (2026).  
> https://doi.org/10.1007/s10489-026-07559-6

---

## Overview

Detecting tiny agricultural pests in natural field imagery is challenging because pests may be small, densely distributed, partially occluded, visually similar to plant structures, and embedded in cluttered backgrounds. ESRepNet is an efficient scaled deep learning framework developed to improve tiny-pest detection while balancing detection quality and inference efficiency.

The framework integrates four principal contributions:

- **A-ELAN (Attention-guided Efficient Layer Aggregation Network):** combines efficient feature aggregation with locally constrained attention to strengthen discriminative local and global cues.
- **MSFF (Multi-Scale Feature Fusion):** combines multi-kernel feature extraction with spatial pyramid pooling and cross-stage partial feature processing to represent pests at different scales.
- **ENST (Efficient Network Scaling Techniques):** uses compound and advanced scaling to balance detection accuracy, throughput, and computational cost.
- **Personalized re-parameterization:** consolidates redundant network components to reduce inference overhead while retaining detection performance.

The network uses a 640 × 640 input resolution in the reported experiments.

## Key results

| Benchmark | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | Inference |
|---|---:|---:|---:|---:|---:|
| COPest | **98.33 ± 0.15%** | **94.67%** | **97.13%** | **84.56%** | — |
| Pest24 | — | — | **78.32 ± 0.12%** | — | **370.4 FPS** |

Additional efficiency findings reported in the paper:

- On Pest24, ESRepNet reaches **370.37 FPS** after personalized re-parameterization, compared with 350.47 FPS before re-parameterization.
- Re-parameterization reduces the reported layer count from **247 to 233** (a 5.7% reduction) without changing the reported mAP.
- The paper reports a **19.9% reduction in inference latency relative to the baseline**.
- The reported computational cost is **105.5 GFLOPs on COPest** and **102.5 GFLOPs on Pest24**.

**Benchmarking note:** FPS values were measured with native PyTorch, FP32 precision, and the batch sizes specified in the paper (16 for COPest and 8 for Pest24), on an NVIDIA GeForce RTX 4090. They should not be interpreted as hardware-independent speed guarantees.

## Datasets

### COPest

The Camellia oleifera Pest (COPest) dataset contains **3,325 images**, **23 pest categories**, and **15,040 annotated pest instances**. It combines:

- 1,306 natural field images collected at Fuzhou Botanical Gardens, Fujian, China;
- 742 images sourced from publicly accessible online repositories;
- 1,277 augmented images.

COPest is available from the first author upon reasonable request, as stated in the article.

### Pest24

Pest24 is a public benchmark containing **25,378 field-captured images** across **24 pest categories**. The dataset is available from its original public source; please consult the paper's data-availability statement and cited dataset reference.

## Method at a glance

```text
Input image (640 × 640)
          |
          v
   Image enhancement
          |
          v
   ESRepNet backbone
   ├── E-ELAN: contextual feature refinement
   └── A-ELAN: attention-guided local/global features
          |
          v
   MSFF: multi-scale feature fusion
   ├── MCFF: parallel 1×1, 3×3, 5×5, 7×7 convolutions
   ├── SPP: multi-level spatial pooling
   └── CSP-style feature processing
          |
          v
   Detection head with D-ELAN
          |
          v
   Multi-scale pest predictions
          |
          v
   Personalized re-parameterization
   and efficient network scaling (ENST)
```

For the detailed network design, please refer to **Figure 3 (overall architecture), Figure 4 (ELAN variants), and Figure 5 (MSFF)** in the paper.

## Repository structure

The repository currently includes the following main components:

```text
ESRepNet/
├── cfg/                    # Model configuration files
├── data/                   # Dataset configuration / data-related files
├── models/                 # Network modules
├── utils/                  # Utility functions
├── train.py                # Training entry point
├── train_aux.py            # Auxiliary training entry point
├── detect.py               # Image/video/stream inference
├── test.py                 # Evaluation entry point
├── export.py               # Model export
├── reparametarization.py   # Re-parameterization utilities
├── generate_heatmap.py     # Heatmap generation
├── visualization.py        # Visualization utilities
└── README.md
```

## Installation

The repository is based on a PyTorch YOLO-style training and inference codebase. The paper reports experiments with **PyTorch 2.1.1** and **CUDA 11.8** on an NVIDIA GeForce RTX 4090.

```bash
git clone https://github.com/suzauddola/ESRepNet.git
cd ESRepNet

# Create and activate an environment (example)
conda create -n esrepnet python=3.10 -y
conda activate esrepnet

# Install a PyTorch build compatible with your CUDA setup.
# Then install the remaining dependencies required by the repository.
```

Please inspect the repository's imports and your local CUDA/PyTorch compatibility before selecting dependency versions. The paper's experimental environment is a reference configuration, not a guarantee that the same setup is required on every machine.

## Dataset preparation

1. Obtain the dataset files and ensure that image and annotation paths are valid.
2. Arrange the data in the directory structure expected by the dataset configuration.
3. Update the corresponding dataset YAML file with the training and validation paths, number of classes, and class names.
4. For COPest access, contact the first author as indicated in the article. Pest24 is publicly available from its original source.

Do not upload restricted or permission-controlled dataset files to this repository.

## Training

`train.py` exposes the configuration, dataset, image-size, epoch, batch-size, and device arguments. Replace the placeholders below with the actual configuration files in your local checkout:

```bash
python train.py \
  --cfg <path-to-model-yaml> \
  --data <path-to-dataset-yaml> \
  --weights <pretrained-checkpoint-or-empty-string> \
  --epochs 300 \
  --batch-size 16 \
  --img-size 640 640 \
  --device 0
```

For a training run from configuration without pretrained weights, use the empty string for `--weights` if that is appropriate for the selected configuration:

```bash
--weights ''
```

The paper used batch size 16 for COPest and 8 for Pest24. Adjust the batch size to fit your GPU memory.

## Inference

`detect.py` accepts a checkpoint, an image or directory (and supported video/stream sources), image size, confidence threshold, IoU threshold, and device:

```bash
python detect.py \
  --weights <path-to-trained-checkpoint.pt> \
  --source <path-to-image-or-image-directory> \
  --img-size 640 \
  --conf-thres 0.25 \
  --iou-thres 0.45 \
  --device 0
```

To save detection labels as text files, add:

```bash
--save-txt
```

The default detection output directory is `runs/detect/`.

## Evaluation

The repository includes `test.py` for model evaluation. Check its command-line arguments and the relevant dataset configuration in your checkout before running an evaluation. For a fair comparison with the paper, keep the dataset split, input resolution, precision mode, batch size, and evaluation protocol consistent with the reported setup.

## Reproducibility and deployment considerations

- Reported speed depends on GPU model, CUDA/PyTorch versions, batch size, precision, and measurement protocol.
- The paper's FPS measurements used native PyTorch and FP32; TensorRT optimization was not included in those reported measurements.
- The paper identifies evaluation on representative edge hardware (e.g., Jetson-class devices or mobile GPUs) and broader testing under unseen field conditions as future work.
- COPest is available from the first author upon reasonable request; access to the dataset should follow the conditions stated in the paper.

## Citation

If you use this work, please cite:

```bibtex
@article{suzauddola2026esrepnet,
  title   = {An efficient scaled deep learning model for detection of pests in complex agricultural environment},
  author  = {Suzauddola, Md and Chen, Junde and Zeb, Adnan and Zhang, Defu},
  journal = {Applied Intelligence},
  volume  = {56},
  article = {562},
  year    = {2026},
  doi     = {10.1007/s10489-026-07559-6}
}
```

## Acknowledgements

The authors acknowledge the research contributions and institutional support described in the published article.

## Contact

**Md Suzauddola**  
Department of Computer Science and Engineering  
United International University, Dhaka, Bangladesh  
Email: [suzauddola@cse.uiu.ac.bd](mailto:suzauddola@cse.uiu.ac.bd)

- **Code:** https://github.com/suzauddola/ESRepNet
- **Paper:** https://doi.org/10.1007/s10489-026-07559-6

## License

Please check the repository for an explicit license before redistributing or using the code in other projects. If you intend this implementation to be openly reusable, consider adding a suitable `LICENSE` file and clarifying the terms of use.
