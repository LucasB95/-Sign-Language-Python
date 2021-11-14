#crear modelo y entrenarlo
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator #ayuda a preprocesar las imagenes
from tensorflow.python.keras import optimizers #optimizador para entrenar el modelo
from tensorflow.python.keras.models import Sequential #nos permite hacer redes neuronales secuenciales
from tensorflow.python.keras.layers import  Dropout, Flatten, Dense #para la creacion de capas neuronales
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D #para hacer capas convoluciones
from tensorflow.python.keras import backend as K #cerras sesiones keras para tener todo limpio cuando se crea la red