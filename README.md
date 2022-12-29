# Handwritten Text Recognition system

* **Graduation project for my Master's degree (In progress)**


* **Based on CNN, BRNN & CTC architectures.**
* **Implemented with TensorFlow (TF).**
* **Trained on the IAM off-line HTR dataset.**
* **The model takes images of single words as input & outputs the recognized text.**
* **And more..(im not good at writing the README)**
----
<a href="#" target="_blank"> <img src="assets/arch.png" alt="tf" width="80%" height="80%"/> </a>

## **How to use it**
Clone it 

```
https://github.com/Moha-boukhatem/HTR.git
```

Create & activate the ENV

```
python3 -m venv venv
. venv/bin/activate
```

Install requirements

```
pip3 install -r requirements.txt
```

## **Preprocess IAM Dataset**

Follow these instructions to get the IAM dataset:

* Register for free at this [website](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database)
* Download `words/words.tgz`
* Download `ascii/words.txt`
* Create a directory for the dataset on your disk, and create two subdirectories: `words` and `ascii`
* Put `words.txt` into the `ascii` directory
* Put the content (directories `a01`, `a02`, ...) of `words.tgz` into the `words` directory
* Change the `path` from `PreprocessFile.CleanData`  with the current `path` of ascii directory
* Change the `base_path` from `PreprocessFile.ImagePathsLabels`  with the current `path` of word directory

Run the CL

```
cd Preprocess
python3 Main.py
```
the Main file will
* Clean & Split data
* Generate New Data Files in New Folder named "Data"

All Preprocess Steps will be demonstrated by some animations in the Terminal



<a href="#" target="_blank"> <img src="assets/Terminal.png" alt="tf" width="80%" height="80%"/> </a>



## **Training Model**

```
model.fit (X, Y, Epochs)
```
* Y = Training data
* X = Validation data
* Epochs = Training iteration


## **Testing Model**

<a href="#" target="_blank"> <img src="assets/new_output1.jpg" alt="tf" width="80%" height="80%"/> </a>

## **Predictions**

<a href="#" target="_blank"> <img src="assets/new_output.jpg" alt="tf" width="80%" height="80%"/> </a>
