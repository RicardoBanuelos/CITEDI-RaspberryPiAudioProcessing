from contextlib import contextmanager
from ctypes import *
from os import system
import concurrent.futures
import math
import matplotlib.pyplot as plt
import numpy as np
import pyaudio 
import struct

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
        return 20*math.log10(rms) + 76

def do_something(frames):
    print(frames)


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

form_1 = pyaudio.paInt16                # Resolucion de 16 bits
chans = 1                               # 1 canal
samp_rate = 44100                       # 44.1 KHz muestras
chunk = 4096                            # 2^12 muestras para el buffer
record_secs = 5                         # Segundos a grabar
dev_index = 0                           # Indice encontrado por p.get_device_info_by_index(i)

# Creamos stream de pyaudio
stream = audio.open(
    format = form_1,
    channels = chans,
    rate = samp_rate,
    input_device_index = dev_index,
    input = True,
    frames_per_buffer = chunk
)

try:
    print("Press Ctrl + C to terminate program.")
    i = 0
    while True:
        dB = 0.0
        frames = []
        for i in range(0, int(samp_rate/chunk)*2):
            data = stream.read(chunk, exception_on_overflow = False)
            frames.append(data)
            rms = get_rms(data)
            dB += rms_to_decibels(rms)
        dB /= int(samp_rate/chunk)
        print("dB: "+str(dB))
        if(dB > 20):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                executor.map(do_something, frames)
        

except KeyboardInterrupt:
    print("\nProgram terminated.")
    # Cierra stream, y termina la instancia de pyaudio
    stream.close()
    audio.terminate()
    pass
