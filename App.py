import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import os
from io import BytesIO

# App
def predictDigit(image):
   try:
       model = tf.keras.models.load_model("model/handwritten.h5")
       image = ImageOps.grayscale(image)
       img = image.resize((28,28))
       img = np.array(img, dtype='float32')
       img = img/255
       img = img.reshape((1,28,28,1))
       pred = model.predict(img)
       result = np.argmax(pred[0])
       return result
   except Exception as e:
       st.error(f"Error en predicción: {str(e)}")
       return None

# Configuración de la página
st.set_page_config(page_title='Reconocimiento de Dígitos escritos a mano', layout='wide')
st.title('Reconocimiento de Dígitos escritos a mano')
st.subheader("Dibuja el digito en el panel y presiona 'Predecir'")

# Parámetros del canvas
drawing_mode = "freedraw"
stroke_width = st.slider('Selecciona el ancho de línea', 1, 30, 15)
stroke_color = '#FFFFFF'
bg_color = '#000000'

# Crear el canvas
canvas_result = st_canvas(
   fill_color="rgba(255, 165, 0, 0.3)",
   stroke_width=stroke_width,
   stroke_color=stroke_color,
   background_color=bg_color,
   height=200,
   width=200,
   key="canvas",
)

# Botón de predicción
if st.button('Predecir'):
   if canvas_result.image_data is not None:
       try:
           # Crear directorio si no existe
           os.makedirs('prediction', exist_ok=True)
           
           # Convertir datos del canvas a imagen
           input_numpy_array = np.array(canvas_result.image_data)
           input_image = Image.fromarray(input_numpy_array.astype('uint8'), 'RGBA')
           
           # Guardar imagen en buffer de memoria
           img_buffer = BytesIO()
           input_image.save(img_buffer, format='PNG')
           img_buffer.seek(0)
           
           # Guardar también en disco por si es necesario
           input_image.save('prediction/img.png')
           
           # Hacer predicción
           img = Image.open(img_buffer)
           res = predictDigit(img)
           
           if res is not None:
               st.header(f'El Dígito es: {str(res)}')
       except Exception as e:
           st.error(f"Error en el procesamiento: {str(e)}")
           st.info("Por favor intenta dibujar nuevamente")
   else:
       st.header('Por favor dibuja en el canvas el dígito.')

# Barra lateral
st.sidebar.title("Acerca de:")
st.sidebar.text("En esta aplicación se evalua ")
st.sidebar.text("la capacidad de un RNA de reconocer") 
st.sidebar.text("digitos escritos a mano.")
st.sidebar.text("Basado en desarrollo de Vinay Uniyal")
