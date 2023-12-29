# Comparative Analysis of Deep Learning Approaches for Earthquake Damage Assessment: A Case Study Incorporating Self-Supervised Contrastive Learning in the 2023 Turkish Earthquakes

## Introduction

This project focuses on the semantic segmentation of building damages resulting from the devastating earthquakes that struck the Kahramanmaraş region in Turkey in February 2023. The earthquakes, with magnitudes of Mw 7.8 and Mw 7.5, led to significant human casualties and extensive infrastructure damage, making the rapid and accurate assessment of building damages critical for effective disaster response and recovery.

To address this challenge, we have leveraged advanced Deep Learning techniques, specifically convolutional neural networks (CNNs) with Early Fusion and Siamese architectures. What sets our approach apart is the incorporation of self-supervised learning through Siamese contrastive learning. This innovative approach enhances our models' ability to detect both subtle and significant alterations in the urban fabric caused by the seismic events.

One of the primary objectives of this project is to provide a comprehensive understanding of the extent and nature of building damages. We have meticulously labeled a dataset based on high-resolution optical satellite imagery captured before and after the earthquakes in the Kahramanmaraş region. This dataset serves as a valuable resource for training and evaluating our models.

In this README, you will find instructions for setting up the project environment, running the code, and accessing the dataset.

## Repository Contents

* [📁 config](config): Contains configuration files for the models.
* [📁 data](data): Contains the dataset and the pretrained model.
* [📁 documents](documents): Contains the project report.
* [📁 notebooks](notebooks): Contains the Jupyter notebook for running the inference.
* [📁 scripts](reports): Contains the project report.
* [📁 src](src): Contains the source code for the models.
* [📄 environment.yml](environment.yml): Contains the Conda environment file.

## Installation Guide

Before beginning, ensure you have the following installed:

* Python (version 3.11.5)
* Conda (version 23.7.4)
* Git (for cloning the repository)

### Cloning the Repository

Clone the repository to your local machine using Git:

```bash
git clone https://github.com/ozan-guven/env-540-earthquake-project/
cd env-540-earthquake-project
``` 

### Environment Setup

It is recommended to create a new Conda environment for this project to manage dependencies effectively. Use the provided [environment.yml](/environment.yml) file to create the environment:

```bash
conda env create -f environment.yml
conda activate dd
```

## Running the Code

1. **Download the Pretrained Model:**

   You can download the pretrained model from this link: [siameseunetdiff.pth](https://github.com/ozan-guven/env-540-earthquake-project/blob/main/data/best_model/siameseunetdiff.pth). After downloading it, place the file in the `data/best_model` folder of your local repository.

2. **Download the Data:**

   Download the data from this link: [damaged.zip](https://github.com/ozan-guven/env-540-earthquake-project/blob/main/data/damaged.zip). Once downloaded, unzip the contents and place them in the `data` folder of your local repository.

   The structure of the project directory should look like this:

   ```bash
    env-540-earthquake-project/
    │
    ├── data/
    │   ├── best_model/
    │   │   └── siameseunetdiff.pth
    │   │
    │   └── damaged/
    │       ├── masks/
    │       ├── post/
    │       └── pre/
    │
    ├── notebooks/
    │   └── inference.ipynb
    │
    ├── environment.yml
    ├── README.md
    └── ...
    ```

3. **Run the Inference Notebook:**

   Open the Jupyter notebook named [inference.ipynb](notebooks/inference.ipynb) located in the project directory. Run the cells within the notebook to perform earthquake damage assessment using the pretrained model and the downloaded data.
