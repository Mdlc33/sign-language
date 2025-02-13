import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import keras

# Cargar el modelo
load_model_2=keras.models.load_model('modelo_Sign_Language.h5')


# Crear intefaz de usuario

st.title('SIGN Language')
st.write('Sube una imagen de un numero en lenguaje de signos')
#subir imagen
uploaded_file = st.file_uploader("Sube la imagen", type=["jpg", "jpeg", "png"])


if uploaded_file is not None:

  imagen_original=Image.open(uploaded_file)
  im=np.asarray(imagen_original.resize((100,100)))
  im=im.reshape(1,100,100,3)

  # Mostrar la imagen subida

  st.image(imagen_original,caption='Imagen subida',use_column_width=True)

  # prediccion
  prediction=load_model_2.predict(im)

  classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

  st.write('Predicción:', classes[np.argmax(prediction)])
