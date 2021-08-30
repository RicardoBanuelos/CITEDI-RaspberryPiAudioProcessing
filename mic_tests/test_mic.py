# Codigo tomado de https://makersportal.com/blog/2018/8/23/recording-audio-on-the-raspberry-pi-with-python-and-a-usb-microphone

from contextlib import contextmanager
from ctypes import *
from os import system
import math
import numpy as np
import pyaudio 
import struct
import wave

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

for i in range(audio.get_device_count()):
    print(audio.get_device_info_by_index(i))

dev_index = int(input("Ingresa el indice de tu dispositivo: "))

form_1 = pyaudio.paInt16                # Resolucion de 16 bits
chans = 1                               # 1 canal
samp_rate = 16000                       # 44.1 KHz muestras
record_secs = 2                         # Segundos a grabar
chunk = samp_rate*record_secs           # Muestras a grabar
wav_output_filename = "test1.wav"       # Nombre del archivo wav que vamos a generar

# Instancia que va a leer del Mini USB Microphone
stream = audio.open(
    format = form_1,
    rate = samp_rate,
    channels = chans, 
    input_device_index = dev_index,
    input = True,
    frames_per_buffer = chunk
)

print("Grabando...")
data = stream.read(chunk, exception_on_overflow = False)
print("Fin de grabacion.")
rms = get_rms(data)
dB = rms_to_decibels(rms)
print(dB)
frames = np.frombuffer(data, dtype="int16")


# Detenemos instancia de lectura, la cerramos y terminamos instancia de PyAudio
stream.stop_stream()
stream.close()
audio.terminate()

# Guardamos los frames del audio como archivo.wav

wavefile = wave.open(wav_output_filename, "wb")
wavefile.setnchannels(chans)
wavefile.setsampwidth(audio.get_sample_size(form_1))
wavefile.setframerate(samp_rate)
wavefile.writeframes(b''.join(frames))
wavefile.close()





