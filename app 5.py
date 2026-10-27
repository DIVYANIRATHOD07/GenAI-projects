from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import base64
import io

from PIL import Image


app = Flask(__name__)



# Load the trained VAE models


DECODER_PATH = "models/vae_decoder.keras"
ENCODER_PATH = "models/vae_encoder.keras"

decoder = tf.keras.models.load_model(DECODER_PATH)
encoder = tf.keras.models.load_model(ENCODER_PATH)

print("VAE Decoder loaded successfully!")
print("Decoder input shape:", decoder.input_shape)
print("Decoder output shape:", decoder.output_shape)
print("VAE Encoder loaded successfully!")
print("Encoder input shape:", encoder.input_shape)

# Build class-aware latent centers from MNIST to generate accurate digit-specific ops
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
all_x = np.concatenate([x_train, x_test], axis=0).astype(np.float32) / 255.0
all_x = all_x[..., np.newaxis]
all_y = np.concatenate([y_train, y_test], axis=0)

z_mean, _ = encoder.predict(all_x, verbose=0)
LATENT_CENTERS = np.stack([
    z_mean[all_y == digit].mean(axis=0)
    for digit in range(10)
], axis=0)

# Preserve each digit's learned latent spread so every sample can vary
# while remaining close to the selected class.
LATENT_SCALES = np.stack([
    z_mean[all_y == digit].std(axis=0)
    for digit in range(10)
], axis=0)

print("Latent centers shape:", LATENT_CENTERS.shape)


# --------------------------------------------------
# Home Page
# --------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


# --------------------------------------------------
# Generate Image
# --------------------------------------------------

@app.route("/generate", methods=["POST"])
def generate():

    try:

        # Get selected digit from UI
        data = request.get_json()

        selected_digit = int(data.get("digit", 0))

        # Validate digit
        if selected_digit < 0 or selected_digit > 9:
            return jsonify({
                "success": False,
                "error": "Please select a number between 0 and 9."
            }), 400


        # --------------------------------------------------
        # Sample around the selected class center so repeated requests produce
        # different handwriting while staying in the class's latent region.
        # --------------------------------------------------

        z = np.random.normal(
            loc=LATENT_CENTERS[selected_digit],
            scale=LATENT_SCALES[selected_digit] * 0.7
        ).reshape(1, -1).astype(np.float32)

        # --------------------------------------------------
        # Generate image using decoder
        # --------------------------------------------------

        generated_image = decoder.predict(
            z,
            verbose=0
        )[0]


        # Remove channel dimension
        generated_image = generated_image.squeeze()


        # Convert values from 0-1 to 0-255
        generated_image = (
            generated_image * 255
        ).astype(np.uint8)


        # --------------------------------------------------
        # Resize to a larger display size before encoding
        # --------------------------------------------------

        image = Image.fromarray(
            generated_image,
            mode="L"
        ).resize(
            (280, 280),
            Image.Resampling.LANCZOS
        )

        buffer = io.BytesIO()

        image.save(
            buffer,
            format="PNG"
        )

        buffer.seek(0)


        # Convert image to Base64
        image_base64 = base64.b64encode(
            buffer.getvalue()
        ).decode("utf-8")


        return jsonify({
            "success": True,
            "selected_digit": selected_digit,
            "image": image_base64
        })


    except Exception as e:

        print("ERROR:", str(e))

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# --------------------------------------------------
# Run Flask Application
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )