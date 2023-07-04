# By Luciano Báez lucianobaez@outlook.com
# Código para detectar objetos en un streaming de de video de protocolo RTSP
# Code to detect objects in a video streaming of RTSP protocol
#
# Version 4 - posibilidad de cargar un json con varias zonas de interes
#
# Tenemos que tener instalada la Librería OpenCV y luego instalar numpy
# We have to have the OpenCV Library installed and then install numpy
#


import cv2
import sys
import numpy as np
import json
import os



#Función para cargar los seteos de cámara desde un json
# Function to load camera settings from a json file
def json_load_camera(camerajsonfile):
    camaradic={}
    camaradic["id"]=''
    camaradic["name"]=''
    camaradic["user"]=''
    camaradic["password"]=''
    camaradic["address"]=''
    camaradic["port"]=''
    camaradic["path"]=''
    camaradic["description"]=''
    if os.path.exists(camerajsonfile):
        with open(camerajsonfile) as archivo_camara:
            camaradic =json.load(archivo_camara)
    camaradic["url"]="rtsp://"+camaradic["user"]+":"+camaradic["password"]+"@"+camaradic["address"]+":"+camaradic["port"]+camaradic["path"]
    return camaradic

# Función para cargar las zonas de detección (o ROI en ingles Region de interés) para la camara desde un archivo json. 
# Function to load the Detection zones (or region of interest ROI) for a camera from a json file
def json_load_zones(zonesjsonfile):
    detectionarray=[]
    detectiondic={}
    detectiondic["x"]=0
    detectiondic["y"]=0
    detectiondic["width"]=0
    detectiondic["height"]=0
    detectiondic["labels"]=[]
    detectiondic["description"]=''
    detectiondic["object_in_roi"]= False
    if os.path.exists(zonesjsonfile):
        with open(zonesjsonfile) as archivo_deteccion:
            detectionarray =json.load(archivo_deteccion)
            for detdic in detectionarray:
                detdic["object_in_roi"]= False
    print(detectionarray)
    return detectionarray

# Función para cargar desde un archivo json la configuracion general
# Function to load the general configuration from a json file
def load_rstp_config_json_file(jsonfile):
    mainccfg={}
    mainccfg["dnncfg_filename"]=''
    mainccfg["dnnweights_filename"]=''
    mainccfg["dnnclasesnames_filename"]=''
    mainccfg["camera_jsonfile"]=''
    mainccfg["zones_jsonfile"]=''
    if os.path.exists(jsonfile):
        with open(jsonfile) as jsonmainfile:
            mainccfg =json.load(jsonmainfile)     
    return(mainccfg)


# Procesar Argumentos
# Process Arguments
print("\nargumentos: ", sys.argv[0])
n = len(sys.argv)
if n>2:
    print("\nYou only need to giveme one argument (the rstp config json main file ). And you provided this: ", sys.argv[0])
    print('')
    sys.exit()
if n<2:
    print("\n Please provide the rstp config json main file. ")
    print('')
    sys.exit()

rstp_config_json_file=sys.argv[1]
rstp_config=load_rstp_config_json_file(rstp_config_json_file)
print(rstp_config)

# Cargar los archivos de configuración y pesos pre-entrenados indicados en el diccionario rstp_config
# Load the configuration files and pre-trained weights indicated in the rstp_config dictionary
net = cv2.dnn.readNet(rstp_config["dnncfg_filename"], rstp_config["dnnweights_filename"])

# Obtener los nombres de las clases
# Get the classes names 
classes = []
with open(rstp_config["dnnclasesnames_filename"], "r") as f:
    classes = [line.strip() for line in f.readlines()]



# Configurar capa de salida
# Configure output layer
layer_names = net.getLayerNames()
#output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
output_layers = net.getUnconnectedOutLayersNames()


# Configurar la captura de video desde el streaming de RTSP
# Configure RTSP streaming video capture 
#stream_url = "rtsp://tu_url_del_streaming"
camara=json_load_camera(rstp_config["camera_jsonfile"])

# Cargar las zonas de interes
# Load the areas of interest
roi_array=json_load_zones(rstp_config["zones_jsonfile"])

stream_url = camara["url"]
cap = cv2.VideoCapture(stream_url)

while True:
    # Leer el siguiente frame del streaming
    # Read the next stream frame 
    ret, frame = cap.read()

    if ret:
        # Redimensionar el frame para un procesamiento más rápido
        # Resize the frame for faster processing
        frame = cv2.resize(frame, None, fx=0.4, fy=0.4)

        # Obtener las dimensiones del frame
        # Get the frame dimensions
        height, width, channels = frame.shape

        # Detectar objetos utilizando los valores YOLO
        # Detect objects using YOLO values
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Inicializar listas para almacenar la información de detección
        # Initialize lists to store detection information
        class_ids = []
        confidences = []
        boxes = []

        # Analizar los resultados de la detección
        # Analyze the detection results
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                # índiuce de confianza
                # Trust index
                if confidence > 0.5:  
                    # Obtener las coordenadas del objeto detectado
                    # Get the detected object coordinates
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Calcular las coordenadas de la caja delimitadora
                    # Calculate bounding box coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    # Almacenar la información de detección
                    # Store detection information
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        # Aplicar supresión de no máximos para eliminar detecciones redundantes
        # Apply non-maximum suppression to remove redundant detections
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # Verificar si hay un objeto reconocido solo en la región de interés
        # Check if there is a recognized object only in the region of interest
        object_in_roi = False
        for roi in roi_array:
            roi["object_in_roi"]= False
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = classes[class_ids[i]]

                for roi in roi_array:
                    if label in roi["labels"]:
                        if x >= roi['x']  and y >= roi['y']  and x + w <= roi['x']  + roi['width']  and y + h <= roi['y'] + roi['height']:
                            object_in_roi = True
                            roi["object_in_roi"]= True

                            # Mostrar la coincidencia
                            # Show the match
                            font = cv2.FONT_HERSHEY_PLAIN
                            confidence = confidences[i]
                            color = (roi['detectioncolor']['b'],roi['detectioncolor']['r'],roi['detectioncolor']['r'])  # Color de deteccion
                            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 5), font, 1, color, 2)

                            #break
                if object_in_roi == True:
                   break

        # Dibujar la región de interés en el frame
        # Draw the region of interest in the frame
        #cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (0, 255, 0), 2)
        for roi in roi_array:
            cv2.rectangle(frame, (roi['x'], roi['y']), (roi['x'] + roi['width'], roi['y'] + roi['height']), (roi['areacolor']['b'], roi['areacolor']['g'], roi['areacolor']['r']), 1)
                    
        # Mostrar el frame con las detecciones
        # Show the frame with the detections
        cv2.imshow("detection de vehiculos", frame)

        # Mostrar mensaje si hay un objeto en la región de interés
        # Show message if there is an object in the region of interest
        for roi in roi_array:
            if roi["object_in_roi"]:
                print(roi["message"])

        # Salir si se presiona la tecla 'q'
        # Exit if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Liberar recursos
# Release resources
cap.release()
cv2.destroyAllWindows()
