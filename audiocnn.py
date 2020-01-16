#TSA Software Devlopment Panther Creek High School

import tensorflow as tf
import pickle as pkl
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import numpy as np
import librosa
import librosa.display
import glob
import os

file = open('prediction.txt', 'w')
file1 = open('probs.txt', 'w')
max_pad_len = 174

#Extract Spectrogram from image
def extract_features(file_name):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    pad_width = max_pad_len - mfccs.shape[1]
    mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    print("finished", file_name)
    return mfccs


num_rows = 40
num_columns = 174
num_channels = 1


features = pkl.load(open('features_v2.pkl', 'rb'))
featuresdf = pd.DataFrame(features, columns=['feature','class_label'])
print("Done")

# Convert features and corresponding classification labels into numpy arrays
X = np.array(featuresdf.feature.tolist())
y = np.array(featuresdf.class_label.tolist())

# Encode the classification labels
le = LabelEncoder()
yy = to_categorical(le.fit_transform(y))

# split the dataset
from sklearn.model_selection import train_test_split

#Split data into training and
x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, random_state = 42)

#Reshape spectrogram to 4-d for CNN
x_train = x_train.reshape(x_train.shape[0], num_rows, num_columns, num_channels)
x_test = x_test.reshape(x_test.shape[0], num_rows, num_columns, num_channels)

model = tf.keras.models.load_model('cnnModelP.h5')
#test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
#print(test_loss, test_acc)

#print prediction and class-probabilties from wav file
def print_prediction(file_name):
    arrClass = []
    arrProb = []

    prediction_feature = extract_features(file_name)
    prediction_feature = prediction_feature.reshape(1, num_rows, num_columns, num_channels)

    predicted_vector = model.predict_classes(prediction_feature)
    predicted_class = le.inverse_transform(predicted_vector)
    print(predicted_vector)
    pred_class = predicted_class[0]
    if predicted_vector == [6]:
        pred_class = "Ambient Sound"
    print("The predicted class is:", pred_class, '\n')
    file.write( pred_class + ',')
    predicted_proba_vector = model.predict_proba(prediction_feature)
    predicted_proba = predicted_proba_vector[0]
    for i in range(len(predicted_proba)):
        category = le.inverse_transform(np.array([i]))
        cat = category[0]
        if i == 6:
            cat = "ambient_sound"
        if i == 3:
            cat = "speech/barks"
        arrClass.append(cat)
        arrProb.append(predicted_proba[i] * 100)
        print(cat, "\t\t : ", format(predicted_proba[i], '.32f') )
    return(arrClass, arrProb)



while (True):
    list_of_files = glob.glob('../uploads/*')
    latest_file = max(list_of_files, key=os.path.getctime)
    preds = print_prediction(latest_file)
    file1.write(preds + ',')
