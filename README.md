# 🧠 **Adversarial Attacks on Deep Learning & Computer Vision Models** 📸

## [![Jump to Dataset](https://img.shields.io/badge/JUMP_TO_DATASET-Click_here-blue)](#dataset) [![Jump to Implementation Details](https://img.shields.io/badge/JUMP_TO_IMPLEMENTATION_DETAILS-Click_here-blue)](#implementation-details) [![Jump to Results](https://img.shields.io/badge/JUMP_TO_RESULTS-Click_here-blue)](#results) [![Jump to Key Findings](https://img.shields.io/badge/JUMP_TO_KEY_FINDINGS-Click_here-blue)](#key-findings) [![Jump to Recommendations](https://img.shields.io/badge/JUMP_TO_RECOMMENDATIONS-Click_here-blue)](#recommendations) [![Jump to Key Takeaways](https://img.shields.io/badge/JUMP_TO_KEY_TAKEAWAYS-Click_here-blue)](#key-takeaways) [![Jump to Conclusion](https://img.shields.io/badge/JUMP_TO_CONCLUSION-Click_here-blue)](#conclusion) [![Visit Repository](https://img.shields.io/badge/VISIT_REPOSITORY-GitHub-blue)](https://github.com/your-repo-url) [![Download Dataset](https://img.shields.io/badge/DOWNLOAD_DATASET-MNIST-blue)](http://yann.lecun.com/exdb/mnist/)


[![GitHub Repository](https://img.shields.io/badge/GitHub_Repository-GitHub-blue)](https://github.com/devarchanadev/Advesarial-Attacks-on-Deep-Learning-Computer-Vision-Model)
[![MNIST Dataset Download](https://img.shields.io/badge/MNIST_Dataset_Download-Click_here-blue)](http://yann.lecun.com/exdb/mnist/)

## Table of Contents

- [Overview](#overview)
- [Objectives](#objectives)
- [Business Impact](#business-impact)
- [Tools and Libraries](#tools-and-libraries)
- [Installation](#installation)
- [Dataset](#dataset)
- [Implementation Details](#implementation-details)
- [Results](#results)
- [Key Findings](#key-findings)
- [Recommendations](#recommendations)
- [Key Takeaways](#key-takeaways)
- [Conclusion](#conclusion)
- [Further Reading and Resources](#further-reading-and-resources)
- [Visit My Repository](#visit-my-repository)
- [Download the Dataset](#download-the-dataset)

## Overview

This project explores **adversarial attacks** on **deep learning** and **computer vision** models, specifically targeting a neural network trained on the MNIST dataset. We focus on implementing the **Fast Gradient Sign Method (FGSM)** and **Projected Gradient Descent (PGD)** attacks to understand their impact on model performance and robustness. The pretrained model utilized is `"dacorvo/mnist-mlp"`, available on Huggingface.

## Objectives

- **📊 Dataset Handling:** Load and preprocess the **MNIST dataset** for evaluating **deep learning** models.
- **🤖 Model Loading:** Employ a **pretrained neural network** model from Huggingface's model hub.
- **🔍 Performance Evaluation:** Assess **model accuracy** on clean and adversarially perturbed data.
- **⚔️ Adversarial Attacks:** Implement FGSM and PGD attacks to generate **adversarial examples** and test model resilience.
- **📈 Visualization:** Visualize the **impact of perturbations** on model accuracy and **image quality**.
- **🔬 Analysis:** Examine how varying magnitudes of perturbations affect **model robustness** and **generalization**.

## Business Impact

Understanding and mitigating **adversarial attacks** is critical for deploying robust **deep learning** models in real-world **computer vision** applications. This analysis highlights vulnerabilities, helping businesses enhance **model security** and **reliability**.

## Tools and Libraries

| **Tool/Library** | **Purpose**                                             |
|------------------|---------------------------------------------------------|
| **Transformers** | For loading the pretrained model (`transformers.AutoModel`) |
| **Torchvision**  | For dataset handling and preprocessing (`torchvision.datasets`, `torchvision.transforms`) |
| **Matplotlib**   | For visualizing **results** and **model performance**   |
| **Torch**        | For training, evaluating, and implementing **adversarial attacks** |

## Installation

Set up your environment by installing the required packages:

```bash
pip install transformers torchvision torch matplotlib
```
## Dataset

**MNIST:** A classic **computer vision** dataset of handwritten digits, consisting of 60,000 training images and 10,000 test images, each 28x28 pixels.

## Implementation Details

- **🗂️ Data Loading and Preprocessing:** Transformed the MNIST dataset to be compatible with the **deep learning** model.
- **🔄 Model Loading:** Used `AutoModel.from_pretrained` to load the `"dacorvo/mnist-mlp"` model from Huggingface.
- **⚔️ FGSM and PGD Attacks:** Implemented these methods to evaluate **model robustness** against **adversarial perturbations**.
- **📊 Visualization:** Mapped the effects of different perturbation levels on **model performance** and **visual quality** of perturbed images.

## Results

- **🔺 FGSM Attack:** Increased epsilon values lead to decreased **model accuracy**, showcasing the trade-off between **perturbation magnitude** and **model robustness**.
- **🔵 PGD Attack:** Similar trends where increased epsilon affects accuracy, with **step size** and **iterations** influencing attack effectiveness.
<img width="364" alt="Screenshot 2024-09-02 145929" src="https://github.com/user-attachments/assets/f0359738-e761-43c9-85e9-5970bbf893d7">

## Key Findings

- **📉 Impact of Perturbation:** Higher perturbation magnitudes result in lower **model accuracy**, indicating reduced robustness.
- **🔍 Visual Analysis:** Adversarial perturbations can be subtle, making them hard to detect visually, but they significantly impact **model performance**.
<img width="382" alt="Screenshot 2024-09-02 150224" src="https://github.com/user-attachments/assets/a1b17713-aa6f-493f-a1f7-fe1600e42830">
<img width="383" alt="Screenshot 2024-09-02 151306" src="https://github.com/user-attachments/assets/1b9d0e92-5646-4a9c-a98c-37a85beb74e5">

## Recommendations

- **🔒 Enhance Model Security:** Regularly evaluate models against **adversarial attacks** to improve their robustness.
- **⚖️ Perturbation Analysis:** Use small perturbations to test **model sensitivity** and adjust defenses accordingly.
- **🔄 Continuous Monitoring:** Implement ongoing assessments and updates to maintain **model reliability** and security.

## Key Takeaways

- **🔐 Understanding Adversarial Attacks:** Crucial for developing secure and reliable **deep learning** and **computer vision models**.
- **⚖️ Balancing Perturbations:** Small perturbations can effectively attack models while remaining visually subtle.
- **📝 Evaluation Strategy:** Regular evaluations against various **adversarial techniques** help in identifying and addressing **model vulnerabilities**.

## Conclusion

Examining FGSM and PGD attacks provides essential insights into the vulnerabilities and robustness of **deep learning** models. Understanding these adversarial techniques aids in developing more secure models and implementing effective countermeasures, ensuring better **performance** and **security** in real-world **computer vision applications**.

## Further Reading and Resources

- [Huggingface Model Hub](https://huggingface.co/models) 🌐
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/) 📚
- [Fast Gradient Sign Method (FGSM)](https://arxiv.org/abs/1412.6572) 📄
- [Projected Gradient Descent (PGD)](https://arxiv.org/abs/1706.06083) 📄

## Visit My Repository

- [![GitHub Repository](https://img.shields.io/badge/GitHub_Repository-GitHub-blue)](https://github.com/devarchanadev/Advesarial-Attacks-on-Deep-Learning-Computer-Vision-Model)) 🔗

## Download the Dataset

- [![MNIST Dataset Download](https://img.shields.io/badge/MNIST_Dataset_Download-Click_here-blue)](http://yann.lecun.com/exdb/mnist/) 🔽
