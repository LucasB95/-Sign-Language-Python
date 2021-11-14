import cv2
import mediapipe as mp
import os  #manejo de carpetas


#Creacion de la carpeta donde se almacena el entrenamiento
nombre = 'Mano_Izquierda' #cambiar el nombre y hacer 300 fotos de cada mano
direccion = 'D:\Davinci\Quinto Cuatrimestre\Integracion Tecnologica\TP3\Fotos\Entrenamiento' #Cambiar ruta a validacion
carpeta = direccion + '/' + nombre
if not os.path.exists(carpeta):
    print('Carpeta creada:', carpeta)
    os.makedirs(carpeta)

#Asignamos un contador para el nombre de las fotos
cont = 0

#Leemos la camara
cap = cv2.VideoCapture(0)

#Creamos un objeto que va almacenar la deteccion y el seguimiento de las manos
clase_manos = mp.solutions.hands
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
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                dedos_reg = cv2.resize(dedos_reg, (200,200), interpolation= cv2.INTER_CUBIC) #redimencion de fotos
                cv2.inwrite(carpeta + "/Mano_{}.jpg".format(cont), dedos_reg)
                cont = cont + 1

    cv2.imshow("Video", frame)
    k = cv2.waitKey(1)
    if k == 27 or cont >= 300:
        break
cap.release()
cv2.destroyAllWindows()




