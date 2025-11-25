import os
import io
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, send_file
from ultralytics import YOLO
from PIL import Image
import joblib
import cv2
from flask_cors import CORS

app = Flask(__name__)

# CORS configuration
CORS(app, resources={r"/api/*": {"origins": ["http://localhost:5173", "https://tomatoes-frontend.vercel.app"]}})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- 1. CONFIGURACIÓN Y CARGA DE MODELOS ---
print("🍅 Cargando sistema de IA final... Por favor espera.")

# A. Cargar Segmentador (YOLO)
try:
    yolo_path = os.path.join(BASE_DIR, "best.pt")
    segmentador = YOLO(yolo_path) 
    print("✅ Segmentador YOLO cargado.")
except Exception as e:
    print(f"❌ Error cargando YOLO: {e}")
    segmentador = None

# B. Cargar Modelos de Clasificación
modelos = {}
try:
    cnn_path = os.path.join(BASE_DIR, "modelo_1_cnn.h5")
    modelos['cnn'] = tf.keras.models.load_model(cnn_path, compile=False)
except:
    print("⚠️ Aviso: No se encontró 'modelo_1_cnn.h5'.")

try:
    mobilenet_final_path = os.path.join(BASE_DIR, "modelo_2_mobilenet_final.keras")
    modelos['mobilenet'] = tf.keras.models.load_model(mobilenet_final_path)
    print("✅ Modelo MobileNet final cargado.")
except Exception as e:
    print(f"⚠️ Error con 'modelo_2_mobilenet_final.keras': {e}")
    try:
        mobilenet_old_path = os.path.join(BASE_DIR, "modelo_2_mobilenet.keras")
        modelos['mobilenet'] = tf.keras.models.load_model(mobilenet_old_path)
        print("✅ Modelo MobileNet (versión anterior) cargado.")
    except Exception as e2:
        print(f"❌ No se pudo cargar ningún modelo MobileNet: {e2}")

try:
    svm_path = os.path.join(BASE_DIR, "modelo_3_svm.pkl")
    print("Comprobando existencia SVM:", os.path.exists(svm_path), "->", svm_path)
    modelos['svm'] = joblib.load(svm_path)
    print("✅ Modelo SVM cargado.")
except Exception as e:
    print(f"❌ Error cargando modelo SVM: {e}")

# Lista de clases (debe coincidir con el orden de entrenamiento)
CLASES = ['dañado', 'maduro', 'verde', 'viejo']

# --- 2. RUTAS DEL SERVIDOR ---

# Ruta de salud / ping para React
@app.route('/api/health')
def health():
    return jsonify({"status": "ok", "message": "TomatoGuard API funcionando"}), 200


# (Opcional) Ruta raíz: ya no sirve el index.html, solo un mensaje sencillo
@app.route('/')
def root():
    return jsonify({"message": "Backend TomatoGuard API. Usa las rutas /api/* desde tu frontend React."}), 200


# RUTA 1: Análisis de un solo tomate
@app.route('/api/analizar_uno', methods=['POST'])
def analizar_uno():
    if 'imagen' not in request.files:
        return jsonify({"error": "Falta imagen"}), 400

    nombre_modelo = request.form.get('modelo', 'mobilenet').lower()
    modelo_actual = modelos.get(nombre_modelo)

    if not modelo_actual:
        return jsonify({"error": f"Modelo '{nombre_modelo}' no disponible"}), 400

    try:
        img_pil = Image.open(request.files['imagen']).convert('RGB')
        img_np = np.array(img_pil)
    except:
        return jsonify({"error": "Imagen inválida"}), 400

    # 1. Segmentar con YOLO
    results = segmentador.predict(
        img_np,
        conf=0.25,
        iou=0.7,
        retina_masks=True,
        classes=[0],
        verbose=False
    ) if segmentador else []

    recorte = img_np
    mensaje = "Tomate detectado y recortado."

    if results and len(results[0].boxes) > 0:
        areas = [(box[2]-box[0]) * (box[3]-box[1]) for box in results[0].boxes.xyxy.cpu().numpy().astype(int)]
        idx_mayor = int(np.argmax(areas))
        box = results[0].boxes.xyxy.cpu().numpy()[idx_mayor].astype(int)

        h_img, w_img, _ = img_np.shape
        pad = 10
        x1 = max(0, box[0] - pad)
        y1 = max(0, box[1] - pad)
        x2 = min(w_img, box[2] + pad)
        y2 = min(h_img, box[3] + pad)

        recorte = img_np[y1:y2, x1:x2]
    else:
        mensaje = "⚠️ No se detectó tomate. Usando imagen completa."

    # 2. Clasificar
    recorte_pil = Image.fromarray(recorte).resize((224, 224))
    input_arr = tf.keras.preprocessing.image.img_to_array(recorte_pil)

    if nombre_modelo == 'svm':
        input_svm = input_arr.flatten().reshape(1, -1) / 255.0
        prediccion = modelo_actual.predict_proba(input_svm)
    else:
        prediccion = modelo_actual.predict(np.array([input_arr]))

    probs = prediccion[0]
    idx = int(np.argmax(probs))

    return jsonify({
        "estado": CLASES[idx],
        "confianza": float(probs[idx]),  # mejor devolver número y formatear en React
        "segmentacion": mensaje,
        "detalle": {k: float(v) for k, v in zip(CLASES, probs)}
    }), 200


# RUTA 2: Segmentación Múltiple
@app.route('/api/segmentar_todo', methods=['POST'])
def segmentar_todo():
    if 'imagen' not in request.files:
        return jsonify({"error": "Falta imagen"}), 400

    clasificador = modelos.get('mobilenet')
    if not segmentador or not clasificador:
        return jsonify({"error": "Faltan modelos (YOLO o MobileNet)"}), 500

    file = request.files['imagen']
    img_pil = Image.open(file).convert('RGB')
    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    img_original_np = np.array(img_pil)

    results = segmentador.predict(
        img_original_np,
        conf=0.25,
        iou=0.7,
        retina_masks=True,
        verbose=False
    )
    result = results[0]

    if result.boxes:
        for i, box in enumerate(result.boxes.xyxy.cpu().numpy().astype(int)):
            x1, y1, x2, y2 = box

            if (x2 - x1) < 20 or (y2 - y1) < 20:
                continue

            recorte = img_original_np[y1:y2, x1:x2]
            if recorte.size == 0:
                continue

            recorte_pil = Image.fromarray(recorte).resize((224, 224))
            input_arr = tf.keras.preprocessing.image.img_to_array(recorte_pil)
            pred = clasificador.predict(np.array([input_arr]), verbose=0)
            idx = int(np.argmax(pred[0]))
            clase = CLASES[idx]
            conf = float(pred[0][idx])

            colores = {
                'dañado': (0, 0, 255),
                'verde': (0, 255, 0),
                'maduro': (0, 165, 255),
                'viejo': (10, 10, 10)
            }
            color = colores.get(clase, (255, 255, 255))

            if result.masks:
                try:
                    if result.masks.xy[i].size > 0:
                        contour = result.masks.xy[i].astype(np.int32).reshape(-1, 1, 2)
                        cv2.drawContours(img_bgr, [contour], -1, color, 2)
                except:
                    pass

            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
            label = f"{clase[:3].upper()} {int(conf * 100)}%"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img_bgr, (x1, y1 - 20), (x1 + w, y1), color, -1)
            cv2.putText(img_bgr, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 2)

    img_final_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_io = io.BytesIO()
    Image.fromarray(img_final_rgb).save(img_io, 'JPEG', quality=90)
    img_io.seek(0)

    # Para React es cómodo descargarlo como blob
    return send_file(img_io, mimetype='image/jpeg')


if __name__ == '__main__':
    app.run(debug=True, port=5000)
