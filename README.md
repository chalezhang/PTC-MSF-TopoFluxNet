# PoreAI — Pore-Throat-Conditioned Multi-Scale Topological Fusion Network

This project uses converged flow fields from OpenFOAM Foundation 13 to train neural networks. It compares a **standard 32-channel U-Net** with the **32-channel PTC-MSF-TopoFluxNet**. Both models use exactly the same 12-channel geometric features, harmonic pressure prior, OpenFOAM reference labels, loss functions, and training strategy.

## Architecture Design

| Module | Standard U-Net | PTC-MSF-TopoFluxNet |
|---|---|---|
| Multi-scale CNN | Conventional encoder–decoder architecture | Residual encoder and global bottleneck attention |
| High-resolution pore-throat branch | Not included | Preserves local information about narrow pore throats and walls |
| Pore-Throat-Conditioned Attention (PTCA) | Not included | Uses wall distance, throat weights, connected regions, and geometric harmonic-pressure gradients as conditioning inputs to enhance features at each scale |
| Multi-scale feature fusion | Conventional same-scale skip connections | Three-way spatial Softmax fusion of current-scale, decoder-scale, and adjacent-scale features |
| Selective dense skip connections | Not included | Dynamically aggregates features from the high-resolution local branch, encoder, and decoder |
| Explicit topology | Not included | Strict four-neighbor connectivity among fluid cells, hydraulically motivated edge weights modulated by pore-throat clearance, and graph message passing |
| Outputs | Kinematic pressure p and velocity components Ux/Uy | The same p/U outputs, with optional export of graph gates, pore-throat-conditioned attention, and cross-scale fusion weights |

Default model parameter counts: standard U-Net ≈ **3,354,915**; PTC-MSF-TopoFluxNet ≈ **3,316,809** (32 channels; default graph depth of 6).

## Environment and Script Execution Order

Ubuntu + OpenFOAM Foundation 13, Python >= 3.10, NumPy, SciPy, PyTorch, and Matplotlib.

```bash
bash 00_setup.sh        # Install dependencies on the first run only
bash run_pipeline.sh    # Clear previous cases, training weights, and results, then run stages 01–07 from scratch
```

Individual script execution order:

| Script | Function | Parallelization |
|---|---|---|
| `01_health.sh` | Check OpenFOAM 13, the Python environment, and system resources | Sequential |
| `02_generate.sh` | Generate CFD geometries and meshes for training, throat-augmented, validation, and test sets | `ProcessPoolExecutor`; 3 independent geometry-generation processes by default |
| `03_collect.sh` | Run OpenFOAM simulations for multiple cases and save converged p/U reference labels | `ThreadPoolExecutor` scheduling; 7 concurrent cases by default, with a budget of 2 threads per case |
| `04_train.sh` | Train both models from scratch using identical data and random seeds | PyTorch CPU multithreading |
| `05_evaluate.sh` | Compute test-set-only metrics and export per-case NPZ/CSV/JSON files | CPU model inference |
| `06_visualize.sh` | Generate per-case prediction and error plots for all training, augmented, validation, and test datasets | **Inference in the main process + 3 independent plotting processes, with tasks processed in batches of 8 cases** |
| `07_report.sh` | Export numerical metrics and paired bootstrap confidence intervals only | NumPy computations |
| `08_final_holdout.sh` | **Optional**: Create a new, independent final test set after the models have been frozen | Same geometry generation, CFD simulation, and plotting procedures as above |

Before execution, `run_pipeline.sh` clears the project directories `cases/`, `data/`, `models/`, and `results/`. **All files from previous runs in these directories will be deleted.** Back up any outputs you need to retain before running the pipeline, and do not store personal files in these four generated directories. `00_setup.sh` does not need to be rerun for every pipeline execution.

## Output Directory Structure

```text
results/
  cfd/<split>/geo_<seed>/...                # CFD output and residual/continuity records
  evaluation/<model>/<split>/geo_<seed>/    # NPZ files and metrics.json for AI vs. OpenFOAM reference comparisons
  evaluation/comparison_summary.json       # Test-set metrics for both models
  predictions/<model>/<split>/geo_<seed>.npz
  figures/
    datasets/<split>/geo_<seed>/
      reference_fields.png                 # Geometry, pore throats, pressure, velocity, etc.
      mesh_grid.png
      velocity_vectors.png
    datasets/augmented_mirror/<split>/geo_<seed>/mirror_y.png
    predictions/<model>/<split>/geo_<seed>/
      prediction_and_error.png            # OpenFOAM references, AI predictions, and errors
      velocity_components.png             # Ux/Uy predictions and differences
      conditioned_fusion_maps.png          # Multi-scale/topological attention (TopoFluxNet only)
    summary/training_*.png
    summary/model_metrics.png
    index.csv                              # Complete figure index
    summary.json
  reports/model_metrics.csv
  reports/paired_comparison.csv
  reports/summary.json
```

All automatically generated reports contain **data and figures only**, in CSV, JSON, and PNG formats.

## Results
