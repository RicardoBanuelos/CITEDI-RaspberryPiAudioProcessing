# La mayoria del codigo fue sacado de: 
# https://makersportal.com/blog/2018/8/23/recording-audio-on-the-raspberry-pi-with-python-and-a-usb-microphone

import sys
import ssl
import time
import json
import datetime
import pyaudio 
# import wave
import struct
import math
import logging, traceback
import paho.mqtt.client as mqtt

IoT_protocol_name = "x-amzn-mqtt-ca"
aws_iot_endpoint = "a5r6qbrsvpf9k-ats.iot.us-east-1.amazonaws.com"
url = "https://{}".format(aws_iot_endpoint)

ca = "/home/pi/Training/aws/AmazonRootCA1.pem" 
cert = "/home/pi/Training/aws/0b5df1387020987a0145ad6eafed2a99f03fd2ea1966da35ff58b1b8f226fc34-certificate.pem.crt.txt"
private = "/home/pi/Training/aws/0b5df1387020987a0145ad6eafed2a99f03fd2ea1966da35ff58b1b8f226fc34-private.pem.key"

print(ca)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(log_format)
logger.addHandler(handler)

form_1 = pyaudio.paInt16            # Resolucion de 16 bits
channels = 1                        # Un canal
sampling_rate = 44100               # 44.1KHz (Tasa de muestreo)
chunk = 4096                        # 2^12 muestras para el buffer
record_seconds = 5                  # Segundos a grabar
device_index = 0                    # Indice del dispositivo de audio
wav_output_filename = 'test1.wav'   # Nombre del archivo .wav

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

def decibels(data):
    count = len(data)/2
    format = "%dh"%(count)
    shorts = struct.unpack( format, data)
    sum_squares = 0.0
    for sample in shorts:
        n = sample * (1.0/32768)
        sum_squares += n*n
    rms = math.sqrt( sum_squares / count )
    return (20*math.log10(rms) + 70) * 2


audio = pyaudio.PyAudio()           

# Creamos stream de pyaudio
stream = audio.open(
    format = form_1,
    channels = channels,
    rate = sampling_rate,
    input_device_index = device_index,
    input = True,
    frames_per_buffer = chunk
)

mqtt_object = {
    "latitude": 32.5391048875214250,
    "longitude": -116.94272323069949,
    "Level": 0.0,
    "sensorName": "Mini USB Microphone"
}

if __name__ == '__main__':
    topic = "sensor/soundTester"

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
            while True:
                db = 0.0
                now = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
                for (i) in range(0, int((sampling_rate/chunk))):
                    data = stream.read(chunk, exception_on_overflow = False)
                    db += decibels(data)
                    print(decibels(data))
                print("Average DB:")
                db /= int(sampling_rate/chunk)
                print(db)
                mqtt_object["Level"] = db
                logger.info("try to publish:{}".format(now))
                queryStringParameters = json.dumps(mqtt_object)
                mqttc.publish(topic, queryStringParameters)
                time.sleep(4)

        except KeyboardInterrupt:
            print("\nProgram terminated.")
            pass
        
    except Exception as e:
        logger.error("exception main()")
        logger.error("e obj:{}".format(vars(e)))
        logger.error("message:{}".format(e.message))
        traceback.print_exc(file=sys.stdout)

    # Termina el stream, cerrarlo, y termina la instancia de pyaudio
    stream.stop_stream()
    stream.close()
    audio.terminate()
