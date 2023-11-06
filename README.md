# Voice-and-diabetes-VOCADIAB
# Screening for type 2 diabetes using voice in the adult population of the United States: Results from the Colive Voice study

## Overview
This repository contains the code and datasets used in the Colive Voice study, which aims to screen for type 2 diabetes (T2D) in the adult population of the United States through voice analysis. The project demonstrates the development of a machine learning voice-based T2D screening tool and evaluates its performance, while stratifying it by age and BMI.

## Repository Contents
- `Data_analysis.ipynb`: Jupyter notebook that contains the evaluation of machine learning models for the T2D screening tool and performance stratification analysis based on age and BMI.
- `classification_cross_validation.py`: Python script for conducting cross-validation of the classification models. This script is imported and used in the `Data_analysis.ipynb` notebook.
- `performance_stratification.py`: Python script for performing performance stratification of the models. This script is also imported and used in the `Data_analysis.ipynb` notebook.
- `vocadiab_females_dataset.pkl`: Pickle file containing the dataset with Hybrid Byol-S embeddings, age, BMI, gender, and ADA score for female participants.
- `vocadiab_males_dataset.pkl`: Pickle file containing the dataset with Hybrid Byol-S embeddings, age, BMI, gender, and ADA score for male participants.

## How to Use
To use this repository, follow these steps:
1. Clone the repository to your local machine.
2. Ensure you have Jupyter Notebook or JupyterLab installed to open the `.ipynb` file.
3. Install the required dependencies listed below.
4. Run the `Data_analysis.ipynb` notebook to perform the analysis and model evaluation while specifying the dataset to be used (for male or female groups).

## Requirements
- Python 3.8.5
- scikit-learn
- pandas 1.4.1
- Numpy 1.23.4

## Data
[Hybrid Byol-S](https://arxiv.org/abs/2203.16637) embeddings are extracted from text audio recordings collected from [Colive Voice](https://www.colivevoice.org/) participants.
The datasets are split by gender to facilitate separate analyses.


---

*Note: This project is for research purposes and the models developed are not intended for clinical use without further validation.*
