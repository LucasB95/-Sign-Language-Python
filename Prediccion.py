import cv2
import mediapipe as mp
import os
import numpy as np
from keras_preprocessing.image import load_img, img_to_array
from keras.models import load_model

modelo = 'D:\Davinci\Quinto Cuatrimestre\Integracion Tecnologica\TP3\Modelo.h5'
peso = 'D:\Davinci\Quinto Cuatrimestre\Integracion Tecnologica\TP3\pesos.h5'
cnn = load_model(modelo) #cargamos el modelo
cnn.load_weights(peso) #cargamos los pesos

direccion = 'D:\Davinci\Quinto Cuatrimestre\Integracion Tecnologica\TP3\Fotos\Validacion'
dire_img = os.listdir(direccion)
print("Nombre: ", dire_img)

#leo la camara
cap = cv2.VideoCapture(0)

#Creamos un objeto que va almacenar la deteccion y seguimiento de las manos
clase_manos = mp.solutions.hands
manos = clase_manos.Hands()
manos = clase_manos.Hands() #->1er param False para que no haga deteccion constante
                            #->2do param numero max de manos
                            #->3er param confianza minima de deteccion
                            #->4to param confianza minima de seguimiento


#Metodo para dibujar las manos
dibujo = mp.solutions.drawing_utils #se dibujan 21 puntos en la manos

while(1):
    ret,frame = cap.read()
    color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    copia = frame.copy()
    resultado = manos.process(color)
    posiciones = [] #coordenadas de los puntos de las manos
    #print(resultado.multi_hand_landmarks) #veo si existe la deteccion

    if resultado.multi_hand_landmarks:
        for mano in resultado.multi_hand_landmarks: #buscamos la mano dentre de la lista de manos del descriptor
            for id, ln in enumerate(mano.landmark): #info de mano por id de punto
                #print(id,ln) #vemos resultados en decimales hay q pasarlo a pixeles
                alto, ancho, c = frame.shape   #extraigo ancho y alto de los fotogramas
                corx, cory = int(ln.x*ancho), int(ln.y*alto) #extraigo ubicacion de punto dependiendo de la mano
                posiciones.append([id, corx, cory])
                dibujo.draw_landmarks(frame, mano, clase_manos.HAND_CONNECTIONS)
            if len(posiciones) != 0:
                pto_i1 = posiciones[4] #ultimo dedo
                pto_i2 = posiciones[20] #ultimo dedo
                pto_i3 = posiciones[12] #dedo medio
                pto_i4 = posiciones[0] #comienzo de mano
                pto_i5 = posiciones[9] #punto central
                x1, y1 = (pto_i5[1]-80), (pto_i5[2]-80) #obtenemos el punto inicial y longitudes
                ancho, alto = (x1+80), (y1+80)
                x2, y2 = x1 + ancho, y1 + alto
                dedos_reg = copia[y1:y2, x1:x2]
                dedos_reg = cv2.resize(dedos_reg, (200, 200), interpolation=cv2.INTER_CUBIC)  # redimencion de fotos
                x = img_to_array(dedos_reg) #convertimos la imagen en matriz
                x = np.expand_dims(x, axis=0) #agregamos nuevo eje
                vector = cnn.predict(x) #arreglo de 2 dimenciones donde va a poner 1 en la clase que se crea correcta
                resultado = vector[0] #[1,0] [0,1]
                respuesta = np.argmax(resultado) #nos entrega el indice del valor mas alto
                if respuesta == 1:
                    print(vector, resultado)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(frame, '{}', format(dire_img[0]), (x1, y1 - 5), 1, 1.3, (0, 0, 255), 1, cv2.LINE_AA)



    cv2.imshow("Video", frame)
    k = cv2.waitKey(1)
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()