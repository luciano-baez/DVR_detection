# By Luciano Báez lucianobaez@outlook.com
# Código para detectar objetos en un streaming de de video de protocolo RTSP
# Code to detect objects in a video streaming of RTSP protocol
#
# Tenemos que tener instalada la Librería OpenCV
# We have to have the OpenCV Library installed
import cv2


# Cargar los archivos de configuración y pesos pre-entrenados de YOLO
# Load YOLO pre-trained weights and configuration files
net = cv2.dnn.readNet("yolov3.cfg", "yolov3.weights")

# Cargar los nombres de las clases desde el archivo
# Load the class names from the file
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Configurar capa de salida
# configure output layer
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Configurar la captura de video desde el streaming de RTSP. Reemplaze con sus datos 
# Configure video capture from RTSP streaming. Replace with your data
dvruser="user"
dvrpass="pass"
dvraddress="ipaddress" # IP o fqdn
dvrport="554" # RTSP port
dvrcamerapath="/h264/ch3/sub/av_stream" #DVR camera path
#
#
stream_url = "rtsp://"+dvruser+":"+dvrpass+"@"+dvraddress+":"+dvrport+dvrcamerapath
cap = cv2.VideoCapture(stream_url)

while True:
    # Leer el siguiente frame del streaming
    # Read the next stream frame
    ret, frame = cap.read()

    if ret:
        # Redimensionar el frame para un procesamiento más rápido
        # Resize the frame for faster precessing
        frame = cv2.resize(frame, None, fx=0.4, fy=0.4)

        # Obtener las dimensiones del frame
        # Get frame sizes
        height, width, channels = frame.shape

        # Detectar objetos utilizando los parametros de YOLO
        # Detect objects using YOLO parameters
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

        # Dibujar las cajas delimitadoras y etiquetas en el frame
        # Draw the frame bounding boxes and labels
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = classes[class_ids[i]]
                confidence = confidences[i]
                # Color de la caja y etiqueta (verde)
                color = (0, 255, 0)  
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 5), font, 1, color, 2)

        # Mostrar el frame con las detecciones
        # Show the frame with the detections
        cv2.imshow("Detección de objetos", frame)

        # Salir si se presiona la tecla 'q'
        # Exit if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Liberar recursos
# Release resources
cap.release()
cv2.destroyAllWindows()