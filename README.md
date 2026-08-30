# Embedding_Extractor
Frozen embeddings from pretrained atomistic foundation models, used as ready-made materials descriptors for multi-objective Bayesian optimization.

## Overview
This repository extracts frozen embeddings from six pretrained atomistic foundation models and uses them as materials descriptors for multi-objective Bayesian optimization. The campaign runs over 21,544 lithium-containing compounds from the Materials Project against three competing objectives: energy above the convex hull (E<sub>h</sub>), electrochemical stability window (ESW), and reaction energy with a representative cathode (E<sub>cat</sub>). All three objectives are precomputed for the whole pool, so the true Pareto frontier is known in advance and every query is answered by table lookup.

## Key Features
- Embeddings taken in a single forward pass from a public checkpoint in inference mode, with no fine-tuning and no property labels: the invariant representation immediately upstream of each model's readout head.
  
- Independent Gaussian-process surrogates, one per objective, refit at every generation.
  
- qNEHVI evaluated directly on the discrete candidate pool, so every proposal is an existing compound rather than a continuous point snapped to its nearest neighbour.

## Installation and how to use
### Prerequisites
- Python 3.11 or higher
  
- PyTorch (compatible with your system)

### Step 1: Install PyTorch

Install PyTorch according to your system configuration from the official website. For CPU:

```
pip install torch
```

For CUDA (GPU): Visit https://pytorch.org/get-started/locally/ and select your configuration.

### Step 2: Install Other Dependencies

```
pip install -r requirements.txt
```

### Alternative: Install All at Once

If you prefer to install everything in one command:

```
pip install torch botorch>=0.18.0 gpytorch>=1.15.0 numpy>=2.0.0 pandas>=2.2.0 openpyxl>=3.1.0
```

Apply PCA to the .npz embedding file of the model you want to benchmark. Then open Notebook/MOBO.ipynb in Jupyter Notebook, set the embedding path to the resulting file, and execute the code. The optimization campaign will run, writing the compounds selected at each generation and their objective values to an Excel file.
The GRACE embedding file is available on request from the contact address below.

## Reference

Seo, Y.H., Lee, B.D., Cho, M.Y., Park, W.B., & Sohn, K.-S.  
Pretrained atomistic foundation models provide ready-made descriptors for multi-objective materials discovery
Sejong University & Sunchon National University

## Contact

Kee-Sun Sohn — kssohn@sejong.ac.kr

## License

This project is licensed under the MIT License.
