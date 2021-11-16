#crear modelo y entrenarlo
import tensorflow.keras.optimizers
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator #ayuda a preprocesar las imagenes
from tensorflow.python.keras import optimizers #optimizador para entrenar el modelo
from tensorflow.python.keras.models import Sequential #nos permite hacer redes neuronales secuenciales
from tensorflow.python.keras.layers import  Dropout, Flatten, Dense #para la creacion de capas neuronales
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D #para hacer capas convoluciones
from tensorflow.python.keras import backend as K #cerras sesiones keras para tener todo limpio cuando se crea la red

K.clear_session()

datos_entrenamiento = 'D:\Davinci\Quinto Cuatrimestre\Integracion Tecnologica\TP3\Fotos\Entrenamiento'
datos_validacion = 'D:\Davinci\Quinto Cuatrimestre\Integracion Tecnologica\TP3\Fotos\Validacion'

#Parametros
iteraciones = 20 #Numero de iteraciones para ajustar el modelo
altura, longitud = 200, 200 #tamaño de las imagenes
batch_size = 1 #numero de imagenes a enviar
pasos = 300/1 #numero de veces a procesar en cada iteracion
pasos_validacion = 300/1 #luego de cada iteracion se valida la anterior
filtrosconv1 = 32
filtrosconv2 = 64 #numero de filtros a aplicar en cada convolucion
tam_filtro1 = (3, 3)
tam_filtro2 = (2, 2) #tamaño de los filtros 1 y 2
tam_pool = (2, 2) #tamaño del filtro en max pooling
clases = 2 #mano abierta y cerrada(5 dedos y 0 dedos)
lr = 0.0005 #ajustes de la red neuronal para acercarse a la solucion optima

#Pre-Procesamiento de las imagenes
preprocesamiento_entre = ImageDataGenerator(
    rescale = 1./255, #pasar pixeles de 0 a 255 / imagen binaria -> 0 a 1
    shear_range = 0.3, #generar imagenes inclinadas para mejorar el entrenamiento
    zoom_range = 0.3, #generar imagenes con zoom
    horizontal_flip = True #invertir imagenes
)

preprocesamiento_vali = ImageDataGenerator(
    rescale = 1./255
)

imagen_entreno = preprocesamiento_entre.flow_from_directory(
    datos_entrenamiento, #va a tomar las fotos
    target_size = (altura, longitud),
    batch_size = batch_size,
    class_mode = 'categorical', #clasificacion categorica por clases
)

imagen_validacion = preprocesamiento_vali.flow_from_directory(
    datos_validacion,
    target_size = (altura, longitud),
    batch_size = batch_size,
    class_mode = 'categorical'
)

#Creamos la red neuronal convolucional
cnn = Sequential() #Red neuronal
#Agregamos filtros con el fin de volver la imagen muy profunda pero pequeña
cnn.add(Convolution2D(filtrosconv1, tam_filtro1, padding = 'same', input_shape = (altura, longitud, 3), activation = 'relu')) #creo la primera capa
#Es una convolucion y realizamos configuracion
cnn.add(MaxPooling2D(pool_size = tam_pool)) #desp de la primera capa vamos a tener una capa max pooling y asignamos el tamaño

cnn.add(Convolution2D(filtrosconv2, tam_filtro2, padding = 'same', activation = 'relu')) #agregamos nueva capa

cnn.add(MaxPooling2D(pool_size = tam_pool))

#Ahora vamos a convertir esa imagen profunda a una plana para tener 1 dimension con la toda la info
cnn.add(Flatten()) #aplano imagen
cnn.add(Dense(256, activation = 'relu')) #asigno 256 neuronas
cnn.add(Dropout(0.5)) #apagamos el 50% de las neuronas en la funcion anterior para no sobreajustar la red
cnn.add(Dense(clases, activation = 'softmax')) #ultima capa, es la que nos dice la probabilidad de una imagen sea de alguna clase u otra

#Agregamos parametros para optimizar el modelo
#Durante el entrenamiento tenga una autoevaluacion que se optimice con Adam y la metrica accuracy
opmizar = tensorflow.keras.optimizers.Adam(learning_rate = lr)
cnn.compile(loss='categorical_crossentropy', optimizer=opmizar, metrics=['accuracy'])

#Entrenamos la red
cnn.fit(imagen_entreno, steps_per_epoch=pasos, epochs=iteraciones, validation_data=imagen_validacion, validation_steps=pasos_validacion,)

#Guardamos el modelo
cnn.save('Modelo.h5')
cnn.save_weights('pesos.h5')









































