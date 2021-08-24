# Codigo tomado de https://makersportal.com/blog/2018/8/23/recording-audio-on-the-raspberry-pi-with-python-and-a-usb-microphone

from contextlib import contextmanager
from ctypes import *
from os import system
import pyaudio 
import wave

def clear():
    _ = system("clear")

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
frames = []

for i in range(0, int((samp_rate/chunk)*record_secs)):
    data = stream.read(chunk, exception_on_overflow = False)
    frames.append(data)

print("Fin de grabacion.")

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





