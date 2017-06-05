# Using neural networks to analyze mainstream and sensational news headlines

## Data

Download all the input files from this [link](https://drive.google.com/open?id=0B7w5W73zZGgONE5pbGRsU3FhZnM)
Make sure the file is in the same directory as the source codes.

## Predict headlines using convolutional neural network

"headline_prediction.py" trains a CNN that predicts if a news headline is being labled as "real"or "fake". The script will generate a graph of training and validation accuracy for each epoche.

The model can also be used to test on some main stream news sources, and plot confidence of each news source as being "real". 

To train the model:
```
python headline_prediction.py
```

To train the model and plot confidence of news sources, use the "s" flag:
```
python headline_prediction.py -s
```

## Generate headliens using recurrent neural network

"headline_generator.py" trains a recurrent neural network that generates headlines that mimics the input headliens.

To run the generator: 
```
python headline_generator.py 
``` 
