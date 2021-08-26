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

np.seterr(divide = 'ignore') 

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

hmm_models = joblib.load("HMMmodels.pkl")

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

# Instancia de PyAudio mas una limpia de la terminal
with no_alsa_err():
    audio = pyaudio.PyAudio()
    clear()

def classify_sound(frames):
    wf = wave.open("frames.wav", "wb")
    wf.setnchannels(1)
    wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
    wf.setframerate(44100)
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

form_1 = pyaudio.paInt16                # Resolucion de 16 bits
chans = 1                               # 1 canal
samp_rate = 44100                       # 44.1 KHz muestras
chunk = 4096                            # 2^12 muestras para el buffer
dev_index = 2                           # Indice encontrado por p.get_device_info_by_index(i)

# Creamos stream de pyaudio
stream = audio.open(
    format = form_1,
    channels = chans,
    rate = samp_rate,
    input_device_index = dev_index,
    input = True,
    frames_per_buffer = chunk
)

if __name__=="__main__":
    print("Recording...")
    try: 
        print("Press Ctrl + C to terminate program.")
        i = 0
        while True: 
            dB = 0.0
            frames = []
            for i in range(0, int(samp_rate/chunk)*2):
                data = stream.read(chunk, exception_on_overflow=False)
                frames.append(data)
                rms = get_rms(data)
                dB += rms_to_decibels(rms)
            dB /= int(samp_rate/chunk)*2
            print("dB: "+str(dB))
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

    