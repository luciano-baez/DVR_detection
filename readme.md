# Reconocimiento de video steaming de un DVR - DVR video steaming recognition

## Resumen (Español)
Este pequeño proyecto fue originado con el fin de poder establecer a futuro una alarma automatica que se dispare cuando alguien estaciona en mi garage, impidiendome entrar o sacar mi vehiculo.
Para ello utilice la libreria OpenCV y NumPy.
Para obtener resultados en forma mas rapida, no entrené a la red neuronal, sino que utilicé una ya entranada.
En este repo van a encorar distintos codigos python que iré realizando para obtener el resultado final; que hoy imagino como una RaspberryPi corriendo un agente en python que dispare una alerta que llegue a mi celular. 

## Summary (English)
This small project was created in order to be able to establish an automatic alarm in the future that goes off when someone parks in my garage, preventing me from entering or taking out my vehicle.
To do this, use the OpenCV library and NumPy.
To get faster results, I didn't train the neural network, but used an already-trained one.
In this repo they are going to decorate different python codes that I will be doing to obtain the final result; that today I imagine as a RaspberryPi running an agent in python that triggers an alert that reaches my cell phone.

## Red Neuronal pre entrenada - Pretrained Neural Network
Aqui estan los links de los 3 archivos necesarios para poder hacer andar la red neuronal.
Here are the links of the 3 files needed to make the neural network work.

[https://pjreddie.com/media/files/yolov3.weights](https://pjreddie.com/media/files/yolov3.weights)
[https://github.com/pjreddie/darknet/blob/master/data/coco.names](https://github.com/pjreddie/darknet/blob/master/data/coco.names)
[https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg)


Este es el enlace de YOLO: Detección de objetos en tiempo real
This is the link of YOLO: Real-Time Object Detection
[https://pjreddie.com/darknet/yolo/](https://pjreddie.com/darknet/yolo/)

## Código - Code
-  rtsp_processing.py
    El archivo rtsp_processing.py es el primer código que escribí, con la idea de poder integrar OpenCV y NumPy y cargar el modelo preentrenado para reconocer vehículos. Es decir una especie de test de viabilidad y tiene muchas cosas codificadas.
    En él tienen que asignar los valores de autenticación de la transmisión rtsp.

    The rtsp_processing.py file is the first code I wrote, with the idea of being able to integrate OpenCV and NumPy and load the pretrained model to recognize vehicles. That is to say, a kind of feasibility test and it has many things codified.
    In it they have to assign the authentication values of the rtsp stream.

- rtsp_processing_parameterized.py
    El archivo rtsp_processing_parameterized.py es un código para detectar lo que esté configurado en los archivos json. La idea es que se pueda usar para identificar diferentes tipos de cosas en diferentes áreas específicas del video y todo es personalizable.
    El código debe llamarse pasando el archivo de configuración general como primer parámetro. Ej: **python rtsp_processing_parameterized.py rtsp_processing_garage.json**
    Dentro del archivo de configuración json principal se configuran los nombres de los demás archivos json específicos (del dvr y de las áreas a monitorear) y los archivos de configuración de la red neuronal.

    The rtsp_processing_parameterized.py file is code to detect whatever is set in the json files. The idea is that it can be used to identify different kinds of things in different specific areas of the video and it's all customizable.
    The code must be called by passing the general configuration file as the first parameter. Ex: **python rtsp_processing_parameterized.py rtsp_processing_garage.json**
    Within the main json configuration file, the names of the other specific json files (the dvr config and  the areas to be scanned) and the neural network configuration files are configured.
