# Readme
The code of TSadv:¡¶Black-box Adversarial Attack on Time Series with Local Perturbations¡·
This project includes a demo dataset "ECG200" with the corresponding pre-trained FCN model checkpoints.
All UCR datasets can be downloaded on http://www.timeseriesclassification.com/.
# Data folder
We stratify the _TEST.txt into _eval.txt and _unseen.txt.  
_TRAIN.txt is the training set to train the target model  
_eval.txt is the data to find top k shapelets and _unseen.txt is the dataset  to be attacked.

## Install requirements.txt
pip install -r requirements.txt
## Train a target model
You can run the train_model.py to train a target model.  
The models.py includes the architecture of FCN and ResNet.  
We already provide the "ECG200" pre-trained model.

## Run TSadv quickly
You can quickly run main.py with the default settings to perform the attack on "ECG200" dataset.  
e.g.: 
    
    python main.py --run_tag=ECG200 
The program can create the results folder automatically. 

## Change the default parameters in main.py to run the code again.
The "parser" in main.py set the default value of parameters.
You can change them .e.g.:

    python main.py --run_tag=ECG200 --magnitude_factor=0.01 --maxitr=80




