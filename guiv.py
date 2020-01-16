import tensorflow as tf
import pickle as pkl
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import numpy as np
import librosa
import librosa.display
import pyaudio
import wave
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tkinter as tk

#Font Definitions
LARGE_FONT = ("Verdana", 12)


class AudioC(tk.Tk):

    def __init__(self, *args, **kwargs):

        tk.Tk.__init__(self, *args, **kwargs)

        container = tk.Frame(self)
        container.pack(side='top', fill='both', expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        frame = StartPage(container, self)
        self.frames[StartPage] = frame
        frame.grid(row=0, column=0, sticky='nsew')
        self.show_frame(StartPage)

    def show_frame(self, cont):

        frame = self.frames[cont]
        frame.tkraise()

class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        label = tk.Label(self, text='Start Page', font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

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
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        print(test_loss, test_acc)

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
                pred_class = "ambient_sound"
            if predicted_vector == [3]:
                pred_class = "speech"
            print("The predicted class is:", pred_class, '\n')
            self.Label = pred_class
            predicted_proba_vector = model.predict_proba(prediction_feature)
            predicted_proba = predicted_proba_vector[0]
            for i in range(len(predicted_proba)):
                category = le.inverse_transform(np.array([i]))
                cat = category[0]
                if i ==6:
                    cat = "ambient_sound"
                if i == 3:
                    cat = "speech"
                arrClass.append(cat)
                arrProb.append(predicted_proba[i])
                print(cat, "\t\t : ", format(predicted_proba[i], '.32f') )
            return(arrClass, arrProb)

        #print prediction from wav file
        def streamlined_pred(file_name):
            prediction_feature = extract_features(file_name)
            prediction_feature = prediction_feature.reshape(1, num_rows, num_columns, num_channels)
            predicted_vector = model.predict_classes(prediction_feature)
            predicted_class = le.inverse_transform(predicted_vector)
            print("prediction:", predicted_class[0], '\n')



        FORMAT = pyaudio.paInt16
        CHANNELS = 2
        RATE = 44100
        CHUNK = 1024
        RECORD_SECONDS = 1
        WAVE_OUTPUT_FILENAME = 'cache_audio.wav'

        def itergraph(i):
            audio = pyaudio.PyAudio()
            audio.get_default_input_device_info()
            # start Recording
            stream = audio.open(format=FORMAT, channels=CHANNELS,
                            rate=RATE, input=True,
                            frames_per_buffer=CHUNK)
            #print("recording...")
            frames = []
            for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK)
                frames.append(data)
            #print("finished recording")
            # stop Recording
            stream.stop_stream()
            stream.close()
            audio.terminate()

            waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
            waveFile.setnchannels(CHANNELS)
            waveFile.setsampwidth(audio.get_sample_size(FORMAT))
            waveFile.setframerate(RATE)
            waveFile.writeframes(b''.join(frames))
            waveFile.close()
            classpreds = print_prediction('cache_audio.wav')
            ax.clear()
            ax.bar(classpreds[0], classpreds[1], align='center')
            #streamlined_pred('file.wav')

        ani = animation.FuncAnimation(fig, itergraph)
        plt.show()

app = AudioC()
app.mainloop()
