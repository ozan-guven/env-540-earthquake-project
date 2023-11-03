# Earthquake Damage Assessment using Multi-modal Learning

## Overview

This project aims to develop a system for rapid assessment of earthquake damage using satellite images. The core idea is to employ an autoencoder for detecting anomalies in post-earthquake images by comparing them against a baseline of undamaged areas and to enhance a segmentation model's accuracy by using these anomalies as additional information. This multi-modal approach utilizes both the raw post-earthquake images and a derived "reconstruction error map" as dual channels to identify damaged areas more effectively. Furthermore, we will incorporate self-supervised learning tasks to improve our model with minimal labeled data.

## Data Preparation

1. **Baseline Image Collection**: Gather satellite images of areas before earthquake damage to serve as a baseline for training the autoencoder.

2. **Post-Earthquake Image Collection**: Collect satellite images from after the earthquakes for assessment.

3. **Autoencoder Training**: Use baseline images to train an autoencoder that learns to reconstruct undamaged areas.

4. **Reconstruction Error Maps**: Pass post-earthquake images through the trained autoencoder to generate reconstruction error maps highlighting potential damages.

## Model Training

1. **Initial Segmentation Model**: Train a state-of-the-art segmentation model (such as U-Net or DeepLab) on a dataset comprising of labeled post-earthquake images and their corresponding reconstruction error maps.

2. **Self-supervised Learning**: Implement self-supervised learning tasks on the segmentation model to refine its performance using unlabeled data.

## Evaluation

1. **Manual Annotation**: Expert annotators will review a subset of the post-earthquake images and mark actual damage, providing ground truth for model evaluation.

2. **Model Testing**: Evaluate the segmentation model's performance using a separate test set of post-earthquake images and manually annotated damage areas.

3. **Refinement**: Iterate on the model training with new annotations and self-supervised tasks to continually improve accuracy.

## 6-Week Plan

### Week 1: Project Initialization

#### Objectives:
- Set up the project repository.
- Define directory structure and coding standards.
- Gather and preprocess baseline satellite images for the autoencoder.

#### Tasks:
- [ ] Initialize the project repository and documentation.
- [ ] Define the directory and coding standards.
- [ ] Collect and preprocess baseline images.

### Week 2-3: Autoencoder Development

#### Objectives:
- Develop and train the autoencoder on undamaged images.
- Generate reconstruction error maps from post-earthquake images.

#### Tasks:
- [ ] Design the autoencoder architecture.
- [ ] Train the autoencoder with baseline images.
- [ ] Create reconstruction error maps using the trained autoencoder.

### Week 4: Segmentation Model Prototyping

#### Objectives:
- Develop the initial segmentation model using available labeled data.
- Integrate the reconstruction error maps into the segmentation model training process.

#### Tasks:
- [ ] Train the initial segmentation model on labeled data and error maps.
- [ ] Set up the multi-modal input pipeline.

### Week 5: Self-supervised Learning Implementation

#### Objectives:
- Design self-supervised learning tasks.
- Incorporate self-supervised learning into the model training loop.

#### Tasks:
- [ ] Develop self-supervised tasks (e.g., rotation prediction, jigsaw puzzles, etc.).
- [ ] Integrate self-supervised tasks with the segmentation model.

### Week 6: Evaluation and Iteration

#### Objectives:
- Evaluate the model performance with the test dataset.
- Identify areas for refinement and conduct additional training rounds.

#### Tasks:
- [ ] Manually annotate a subset of the post-earthquake images.
- [ ] Evaluate the model's damage detection performance.
- [ ] Refine the model using additional labeled data and self-supervised learning.

## Notes

- The plan assumes that post-earthquake images and baseline images are already available by Week 1.
- Additional expert annotation might be required for continuous model improvement.
- Self-supervised learning tasks need to be defined based on the characteristics of the satellite images and the nature of the earthquake damage.

By following this structured plan, we aim to develop a robust system for assisting in rapid earthquake damage assessment, which can significantly aid in timely and effective disaster response and management.
