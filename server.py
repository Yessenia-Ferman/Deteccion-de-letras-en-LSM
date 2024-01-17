# Importar librerias
from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
from Funciones.condicionales import condicionalesLetras
from Funciones.normalizacionCords import obtenerAngulos

# Inicializar mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles

# Realizar nuestra videocaptura
cap = cv2.VideoCapture(0)

# Configurar la resolución de la cámara
wCam, hCam = 1280, 720
cap.set(3, wCam)
cap.set(4, hCam)

# Inicializar la variable fuera del bucle
lectura_actual = 0

# Inicializar la aplicación Flask
app = Flask(__name__)

# Definir la función para generar fotogramas
def gen_frame():
    # Inicializar la variable antes de usarla en el bucle
    pinkY = 0
    lectura_actual = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Procesamiento de video con mediapipe
        height, width, _ = frame.shape
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        with mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.75) as hands:

            results = hands.process(frame_rgb)
            if results.multi_hand_landmarks is not None:
                # Accediendo a los puntos de referencia, de acuerdo a su nombre
                angulosid = obtenerAngulos(results, width, height)[0]

                dedos = []
                # pulgar externo angle
                if angulosid[5] > 125:
                    dedos.append(1)
                else:
                    dedos.append(0)

                # pulgar interno
                if angulosid[4] > 150:
                    dedos.append(1)
                else:
                    dedos.append(0)

                # 4 dedos
                for id in range(0, 4):
                    if angulosid[id] > 90:
                        dedos.append(1)
                    else:
                        dedos.append(0)

                TotalDedos = dedos.count(1)
                condicionalesLetras(dedos, frame)

                pinky = obtenerAngulos(results, width, height)[1]
                pinkY = pinky[1] + pinky[0]
                resta = pinkY - lectura_actual
                lectura_actual = pinkY
                print(abs(resta), pinkY, lectura_actual)

                if dedos == [0, 0, 1, 0, 0, 0]:
                    if abs(resta) > 30:
                        print("jota en movimiento")
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.rectangle(frame, (0, 0), (100, 100), (255, 255, 255), -1)
                        cv2.putText(frame, 'J', (20, 80), font, 3, (0, 0, 0), 2, cv2.LINE_AA)

                # testing--------------------------------------
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())

            # Convertir el fotograma a bytes para la transmisión
            suc, encode = cv2.imencode('.jpg', frame)
            frame = encode.tobytes()

            yield (b'--frame\r\n'
                   b'content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# Ruta principal
@app.route('/')
def index():
    return render_template('Index.html')

# Ruta para el video en tiempo real
@app.route('/video')
def video():
    return Response(gen_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Ejecutar la aplicación web
if __name__ == "__main__":
    app.run(debug=False)
