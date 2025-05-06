import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
import tempfile
from PIL import Image

# Load Model Function - Caches the model to avoid reloading each time
@st.cache_resource()
def load_model():
    model = tf.keras.models.load_model('EfficientNetB0_final.keras')
    return model

# TensorFlow Model Prediction
def model_prediction(test_image_path):
    model = load_model()  # Use cached model
    img = tf.keras.utils.load_img(test_image_path, target_size=(224, 224))
    x = tf.keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    predictions = model.predict(x)
    return np.argmax(predictions)  # Return index of the class with the highest probability

# Main Page
st.header("Welcome to Pokemon Identifier")

# Input Image Upload
test_image = st.file_uploader("Upload your Pokemon Image:")

if test_image is not None:
    # Save the uploaded image to a temporary file and get its path
    with tempfile.NamedTemporaryFile(delete=False, suffix=test_image.name) as tmp_file:
        tmp_file.write(test_image.read())
        temp_file_path = tmp_file.name

    # Predict button
    if st.button("Predict") and test_image is not None:
        with st.spinner("Please Wait.."):
            result_index = model_prediction(temp_file_path)
            
            # Class labels (Assuming these correspond to the model classes)
            class_name = ['Charmander', 'Dugtrio', 'Electrode', 'Exeggutor', 'Horsea', 'Jolteon', 'Kangaskhan', 'Lapras', 
                          'Machamp', 'Mankey', 'Metapod', 'Mew', 'Mewtwo', 'Nidoking', 'Nidoqueen', 'Nidorino', 'Oddish', 
                          'Pikachu', 'Poliwag', 'Poliwhirl', 'Primeape', 'Raichu', 'Rhydon', 'Scyther', 'Snorlax', 'Spearow', 
                          'Squirtle', 'Tangela', 'Tauros', 'Vaporeon', 'Venomoth', 'Venusaur', 'Vileplume', 'Voltorb', 
                          'Vulpix', 'Weezing', 'Wigglytuff', 'Zapdos']
            
            pokemon_name = class_name[result_index]  # Get the name of the Pokémon predicted

        # Displaying the result
        st.success(f"Model predicts it's a {pokemon_name}")

        # Display the input image
        st.image(test_image, caption="Input Image", use_container_width=True)





