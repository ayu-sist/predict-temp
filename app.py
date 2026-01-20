from flask import Flask, render_template, request
import numpy as np
import joblib
import tensorflow as tf

app = Flask(__name__)

# =====================
# Load Model & Scaler
# =====================
MODEL_PATH = 'model/model_suhu.h5'
SCALER_PATH = 'model/scaler_suhu.save'

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
scaler = joblib.load(SCALER_PATH)

# =====================
# Route Utama
# =====================
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = None
    form_data = {}

    if request.method == 'POST':
        try:
            sequence_data = []

            # Simpan input agar tidak hilang di form
            form_data = request.form.to_dict()

            # Ambil data 7 hari × 4 fitur
            for i in range(1, 8):
                temp  = float(form_data[f'temp_{i}'])
                hum   = float(form_data[f'hum_{i}'])
                wind  = float(form_data[f'wind_{i}'])
                press = float(form_data[f'press_{i}'])

                sequence_data.append([temp, hum, wind, press])

            # (7, 4)
            sequence_np = np.array(sequence_data)

            # Normalisasi
            sequence_scaled = scaler.transform(sequence_np)

            # Reshape → (1, 7, 4)
            final_input = sequence_scaled.reshape(1, 7, 4)

            # Prediksi
            prediction_scaled = model.predict(final_input, verbose=0)

            # Inverse hanya suhu (fitur ke-0)
            dummy = np.zeros((1, 4))
            dummy[0, 0] = prediction_scaled[0, 0]

            prediction_actual = scaler.inverse_transform(dummy)
            result = prediction_actual[0, 0]

            prediction_text = f"Prediksi Suhu Besok: {result:.2f} °C"

        except Exception as e:
            prediction_text = f"Terjadi kesalahan: {str(e)}"

    return render_template(
        'index.html',
        prediction=prediction_text,
        form_data=form_data
    )

# =====================
# Run App
# =====================
if __name__ == '__main__':
    app.run(debug=True)
