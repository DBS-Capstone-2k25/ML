from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import traceback

app = FastAPI()

# Load model .h5
model = tf.keras.models.load_model("model/garbage_classification_model.h5")

# Label kelas (harus sama urutannya dengan saat training)
label_map = [
    'battery', 'biological', 'cardboard', 'clothes', 'glass',
    'metal', 'paper', 'plastic', 'shoes', 'trash'
]

IMAGE_SIZE = (224, 224)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Baca isi file gambar
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize(IMAGE_SIZE)

        # Preprocessing
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0).astype("float32")

        # Prediksi
        pred = model.predict(img_array)[0]
        predicted_class_index = int(np.argmax(pred))
        predicted_label = label_map[predicted_class_index]
        confidence = float(pred[predicted_class_index])

        # Hasil lengkap + confidence tiap kelas
        results = [
            {"label": label_map[i], "confidence": float(pred[i])}
            for i in range(len(label_map))
        ]
        results.sort(key=lambda x: x["confidence"], reverse=True)

        return JSONResponse(content={
            "predicted_class": predicted_label,
            "confidence": confidence,
            "prediction": results
        })

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)
