from contextlib import contextmanager
from ctypes import *
from os import system
import concurrent.futures
import math
import numpy as np
import pyaudio 
import struct
import wave

from scipy.io import wavfile
from hmmlearn import hmm
from python_speech_features import mfcc
import joblib

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

def clear():
    _ = system("clear")

# Funcion para calcular RMS y convertir a dB tomada de https://stackoverflow.com/questions/25868428/pyaudio-how-to-check-volume
def get_rms(data):
    count = len(data)/2
    format = "%dh"%(count)
    shorts = struct.unpack( format, data)
    sum_squares = 0.0
    for sample in shorts:
        n = sample * (1.0/32768)
        sum_squares += n*n
    return math.sqrt(sum_squares/count)

def rms_to_decibels(rms):
    if rms == 0:
        return 0
    else:
        return 20*math.log10(rms) + 88

def classify_sound(frames):
    wf = wave.open("frames.wav", "wb")
    wf.setnchannels(chans)
    wf.setsampwidth(audio.get_sample_size(form_1))
    wf.setframerate(samp_rate)
    wf.writeframes(b''.join(frames))
    wf.close()

    sampling_freq, sound = wavfile.read("frames.wav")

    mfcc_features = mfcc(sound, sampling_freq)
    max_score = -99999
    output_label = None

    for item in hmm_models:
        hmm_model, label = item
        score = hmm_model.get_score(mfcc_features)
        if score > max_score:
            max_score = score
            output_label = label
    if output_label == None:
        return "unknown"
    else: 
        return output_label

if __name__=="__main__":
    
    form_1 = pyaudio.paInt16                # Resolucion de 16 bits
    chans = 1                               # 1 canal
    samp_rate = 16000                       # 44.1 KHz muestras
    duration = 2                            # Duracion de la grabacion
    chunk = samp_rate*3                     # Cantidad de muestras

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

    try: 
        print("Press Ctrl + C to terminate program.")
        i = 0
        while True: 
            dB = 0.0
            data = stream.read(chunk, exception_on_overflow=False)
            rms = get_rms(data)
            dB += rms_to_decibels(rms)
            print("dB: "+str(dB))
            frames = np.frombuffer(data, dtype="int16")
            if(dB > 20):
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    prediction = executor.submit(classify_sound, frames)
                    print("Predicted: " + prediction.result())

    except KeyboardInterrupt:
        print("\nProgram terminated.")
        # Cierra stream, y termina la instancia de pyaudio
        stream.close()
        audio.terminate()
        pass

    