import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import os
from io import BytesIO

# Cargar el modelo al inicio
@st.cache_resource  # Esto guarda el modelo en caché
def load_model():
    try:
        # Verifica si el directorio existe
        if not os.path.exists('model'):
            os.makedirs('model')
            st.error("El directorio del modelo no existe. Por favor, asegúrate de que el modelo está en la ubicación correcta.")
            return None
            
        # Verifica si el archivo del modelo existe
        if not os.path.exists('model/handwritten.h5'):
            st.error("No se encuentra el archivo del modelo 'handwritten.h5'. Asegúrate de que está en el directorio 'model'.")
            return None
            
        # Cargar el modelo
        return tf.keras.models.load_model("model/handwritten.h5")
    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
        return None

# Cargar el modelo
model = load_model()

# App
def predictDigit(image):
    try:
        if model is None:
            st.error("No se pudo cargar el modelo. Por favor, verifica la instalación.")
            return None
            
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

# Instrucciones de instalación si el modelo no está cargado
if model is None:
    st.warning("""
    Para que esta aplicación funcione, necesitas:
    1. Crear un directorio 'model' en el directorio raíz
    2. Colocar el archivo 'handwritten.h5' en el directorio 'model'
    3. Asegurarte de que el modelo tiene el formato correcto para TensorFlow
    """)

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
    if model is None:
        st.error("No se puede realizar la predicción sin el modelo cargado.")
    elif canvas_result.image_data is not None:
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
                st.success("Predicción realizada con éxito!")
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

# Información adicional sobre el estado del modelo
st.sidebar.markdown("---")
st.sidebar.subheader("Estado del Sistema:")
if model is not None:
    st.sidebar.success("✅ Modelo cargado correctamente")
else:
    st.sidebar.error("❌ Modelo no cargado")
@st.cache_resource
def load_model():
    try:
        # Aquí puedes poner el código para cargar tu modelo
        # Por ejemplo, si tienes el modelo en GitHub:
        # url = "URL_DE_TU_MODELO"
        # model = tf.keras.models.load_model(tf.keras.utils.get_file("handwritten.h5", url))
        # return model
        pass
    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
        return None
