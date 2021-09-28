from contextlib import contextmanager
from ctypes import *
from os import system
import numpy as np

import pyaudio 
import wave

from scipy.io import wavfile
from hmmlearn import hmm
from python_speech_features import mfcc
import joblib

# PyAudio error handler
np.seterr(divide = 'ignore') 

ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
def py_error_handler(filename, line, function, err, fmt):
    pass

c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)

@contextmanager
def no_alsa_err():
    asound = cdll.LoadLibrary("libasound.so")
    asound.snd_lib_error_set_handler(c_error_handler)
    yield
    asound.snd_lib_error_set_handler(None)

# Class to handle all HMM related processing
class HMMTrainer(object):
    def __init__(self, model_name='GaussianHMM', n_components=4, cov_type='diag', n_iter=1000):
        self.model_name = model_name
        self.n_components = n_components
        self.cov_type = cov_type
        self.n_iter = n_iter
        self.models = []

        if self.model_name == 'GaussianHMM':
            self.model = hmm.GaussianHMM(n_components=self.n_components, 
                    covariance_type=self.cov_type, n_iter=self.n_iter)
        else:
            raise TypeError('Invalid model type')

    # X is a 2D numpy array where each row is 13D
    def train(self, X):
        np.seterr(all='ignore')
        self.models.append(self.model.fit(X))

    # Run the model on input data
    def get_score(self, input_data):
        return self.model.score(input_data)

# Function to clear PyAudio terminal spam
def clear():
    _ = system("clear")

if __name__=="__main__":
    
    form_1 = pyaudio.paInt16
    chans = 1
    samp_rate = 16000
    duration = 2
    chunk = samp_rate*2

    # Instancia de PyAudio mas una limpia de la terminal
    with no_alsa_err():
        audio = pyaudio.PyAudio()
        clear()

    # Codigo para encontrar el indice del dispositivo que buscamos
    for i in range(audio.get_device_count()):
        print(audio.get_device_info_by_index(i))

    dev_index = int(input("Seleccion el indice de tu dispositivo: "))

    # Creamos stream de pyaudio
    stream = audio.open(
        format = form_1,
        channels = chans,
        rate = samp_rate,
        input_device_index = dev_index,
        input = True,
        frames_per_buffer = chunk
    )

    hmm_models = joblib.load("HMMmodels.pkl")

    print("Grabando")
    data = stream.read(chunk, exception_on_overflow=False)
    frames = np.frombuffer(data, dtype="int16")
    print("Fin")

    wf = wave.open("classify.wav", "wb")
    wf.setnchannels(chans)
    wf.setsampwidth(audio.get_sample_size(form_1))
    wf.setframerate(samp_rate)
    wf.writeframes(b''.join(frames))
    wf.close()

    sampling_freq, sound = wavfile.read("classify.wav")

    mfcc_features = mfcc(sound, sampling_freq)
    max_score = -99999
    output_label = None

    for item in hmm_models:
        hmm_model, label = item
        score = hmm_model.get_score(mfcc_features)
        if score > max_score:
            max_score = score
            output_label = label

    print("Predicted: " + output_label)

    stream.close()
    audio.terminate()
