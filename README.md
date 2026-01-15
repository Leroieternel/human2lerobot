# Universal Dataset → LeRobot Dataset Conversion Toolkit

This repository provides a **general-purpose data processing and conversion framework** for transforming
**heterogeneous human and robot datasets** into the standardized **LeRobot dataset format**.

It is designed to support:
- Human demonstration datasets
- Robot trajectory datasets
- Egocentric video datasets
- Multimodal datasets with vision, state, action, and language
- Multiple raw data formats (e.g., HDF5, MP4, NumPy, Parquet)

The goal is to make diverse datasets **plug-and-play compatible** with the
[LeRobot](https://github.com/huggingface/lerobot) ecosystem and downstream robot learning pipelines.

This repository addresses that problem by providing a **unified conversion pipeline** that maps
arbitrary raw datasets into the **LeRobot canonical dataset structure**, which has emerged as a de facto standard for training modern vision-language-action (VLA) models.

---

## Motivation

Robot learning datasets come in many forms:

- Human egocentric video + pose (e.g., EgoDex)
- Robot demonstrations stored in HDF5
- Simulation rollouts in NumPy / Parquet
- Mixed vision-language-action datasets
- Custom lab-specific logging formats

Each dataset typically requires **bespoke loading logic**, making it difficult to:
- Reuse training code
- Compare models across datasets
- Scale to large, heterogeneous data sources



---

## What This Repository Does

This toolkit focuses on **data conversion, not model training**.

It provides:
- Parsers for different raw data formats
- Dataset-specific adapters
- Feature extraction and normalization utilities
- Episode construction logic
- Exporters to LeRobot-compatible datasets

After conversion, all datasets share:
- A unified episodic structure
- Standardized observation/action/state representations
- Common metadata and statistics format
- Compatibility with LeRobot dataloaders and tooling

---

## Supported Data Types

### Data Sources
- Human demonstrations
- Robot teleoperation data
- Autonomous robot rollouts
- Simulation trajectories

### Modalities
- RGB / multi-view images
- Proprioceptive states
- Actions
- Language annotations
- 3D pose / kinematics

### Input Formats
- HDF5
- MP4 / video files
- NumPy arrays
- Parquet files
- JSON metadata
- Custom binary formats (via adapters)

---

## Target Format: LeRobot

All processed datasets are exported to the **LeRobot dataset format**, which provides:

- Episodic organization
- Explicit separation of observations, actions, and metadata
- Support for vision, state, action, and language
- Efficient indexing and statistics
- Compatibility with HuggingFace datasets and PyTorch

A converted dataset typically looks like:

