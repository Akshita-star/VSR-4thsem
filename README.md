# Visual Speech Recognition using Deep Learning

## Project Overview

Visual Speech Recognition (VSR) is an AI-based system that converts **lip movements captured from live webcam video into readable text** without relying on audio input.
This project combines **computer vision** and **deep learning** to recognize silent speech and enable communication in **noisy, restricted, or assistive environments**.

A significant portion (**~80%**) of the training data used in this project is **self-created**, ensuring originality, customization, and better control over model training conditions.

---

## Objectives

* Capture **real-time lip movement** using a webcam.
* Build a **self-generated visual speech dataset**.
* Extract meaningful **spatio-temporal features** from lip regions.
* Train a **deep learning model** to convert silent speech into text.
* Demonstrate a **working real-time prototype**.

---

### Libraries & Frameworks

* OpenCV – webcam capture, face & lip detection
* NumPy & Pandas – preprocessing and data handling
* TensorFlow / PyTorch – deep learning model training
* Keras – neural network design
* Matplotlib / Seaborn – visualization and analysis

### Deep Learning Techniques

* Convolutional Neural Networks (CNN) – spatial feature extraction
* Recurrent Networks (LSTM/GRU) – temporal speech understanding
* CTC / Sequence Modeling – frame-to-text prediction

---

##  System Workflow

1. **Live Video Acquisition**
   Real-time lip movement captured through **webcam input**.

2. **Dataset Creation (Self-Generated ~80%)**
   * Recording multiple speakers and words
   * Frame extraction and labeling
   * Data cleaning and normalization

3. **Preprocessing**
   * Face and lip region detection
   * Image resizing and normalization
   * Sequence formation from frames

4. **Feature Extraction & Learning**
   * CNN extracts spatial lip features
   * LSTM/GRU learns temporal speech patterns

5. **Real-Time Text Prediction**
   * Model converts live lip motion into **displayed text output**.

---

## Expected Outcomes

* Functional **real-time webcam-based lip reading system**
* Demonstration of **self-collected dataset training**
* Accurate **silent speech-to-text prediction prototype**


## Author

**Akshita Panwar**
B.Tech CSE (AI/ML)
Project: (VSR:Visual Speech Recognition

-
