from contextlib import contextmanager
from ctypes import *
from os import system
import concurrent.futures
import datetime
import json
import math
import numpy as np
import logging, traceback
import paho.mqtt.client as mqtt
import pyaudio 
import ssl
import sys
import struct
import time

IoT_protocol_name = "x-amzn-mqtt-ca"
aws_iot_endpoint = "a19mplqw6xosam-ats.iot.us-east-1.amazonaws.com"
url = "https://{}".format(aws_iot_endpoint)

ca = "/home/pi/CITEDI-RaspberryPiAudioProcessing/mic_tests/aws/AmazonRootCA1.pem" 
cert = "/home/pi/CITEDI-RaspberryPiAudioProcessing/mic_tests/aws/fb82195cfab8075053229cf7cf0cb3100546168d88f62f232bffb72aa3b4d5ca-certificate.pem.crt"
private = "/home/pi/CITEDI-RaspberryPiAudioProcessing/mic_tests/aws/fb82195cfab8075053229cf7cf0cb3100546168d88f62f232bffb72aa3b4d5ca-private.pem.key"

print(ca)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(log_format)
logger.addHandler(handler)

form_1 = pyaudio.paInt16                # Resolucion de 16 bits
chans = 1                               # 1 canal
samp_rate = 44100                       # 44.1 KHz muestras
duration = 2                            # Duracion de la grabacion
chunk = samp_rate*3                     # Cantidad de muestras

np.seterr(divide = 'ignore') 

def clear():
    _ = system("clear")

def ssl_alpn():
    try:
        #debug print opnessl version
        logger.info("open ssl version:{}".format(ssl.OPENSSL_VERSION))
        ssl_context = ssl.create_default_context()
        ssl_context.set_alpn_protocols([IoT_protocol_name])
        ssl_context.load_verify_locations(cafile=ca)
        ssl_context.load_cert_chain(certfile=cert, keyfile=private)
        return ssl_context

    except Exception as e:
        print("exception ssl_alpn()")
        raise e

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

def do_something():
    return 'Hola Mundo'


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

mqtt_object = {
    "deviceId": "UABC12",
    "soundIntensity": 0.0,
}

if __name__ == '__main__':
    topic = "aws/soundDevices"

    try:
        mqttc = mqtt.Client()
        ssl_context= ssl_alpn()
        mqttc.tls_set_context(context=ssl_context)
        logger.info("start connect")
        mqttc.connect(aws_iot_endpoint, port=443)
        logger.info("connect success")
        mqttc.loop_start()

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
                mqtt_object["soundIntensity"] = dB
                now = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
                logger.info("try to publish:{}".format(now))
                queryStringParameters = json.dumps(mqtt_object)
                mqttc.publish(topic, queryStringParameters)
                if(dB > 20):
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        prediction = executor.submit(do_something)
                        print(prediction.result())
                
        except KeyboardInterrupt:
            print("\nProgram terminated.")
            pass
            
    except Exception as e:
        logger.error("exception main()")
        logger.error("e obj:{}".format(vars(e)))
        logger.error("message:{}".format(e.message))
        traceback.print_exc(file=sys.stdout)
    
    # Cierra stream, y termina la instancia de pyaudio
    stream.close()
    audio.terminate()
