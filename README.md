## Google - Isolated Sign Language Recognition 11th place solution
This repository contains the code to reproduce 11th place result in Kaggle American sign language recognition competition.  
https://www.kaggle.com/competitions/asl-signs

## Solution on Kaggle
Please refer below link for the detailed write up about solution.  
https://www.kaggle.com/competitions/asl-signs/discussion/406657

## 1. Setup environment
```
docker run --gpus all --shm-size 32G --name kaggle gcr.io/kaggle-gpu-images/python /bin/bash
git clone https://github.com/bamps53/kaggle-asl-11th-place-solution
cd kaggle-asl-11th-place-solution
pip install -r requirements.txt 
```

## 2. Prepare data
```
mkdir ../input
cd ../input
kaggle competitions download -c asl-signs
unzip -q asl-signs.zip -d input/asl-signs
cd ../kaggle-asl-11th-place-solution
bash ./prepare_data.sh
```
## 3. Main
```
bash ./run.sh
```
## 4. Convert to TFLite
```
bash ./convert.sh
```