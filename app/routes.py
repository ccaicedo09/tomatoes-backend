import io
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
from flask import Blueprint, current_app, request, jsonify, send_file

api_bp = Blueprint("api", __name__, url_prefix="/api")

@api_bp.route("/health")
def health():
    return jsonify({"status": "ok", "message": "TomatoGuard API funcionando"}), 200

@api_bp.route("/db-test")
def db_test():
    db = current_app.config["MONGO_DB"]
    # Insert test doc
    result = db["tests"].insert_one({"mensaje": "Hola Mongo desde TomatoGuard"})
    # Read it
    doc = db["tests"].find_one({"_id": result.inserted_id})
    doc["_id"] = str(doc["_id"])
    
    return jsonify({"insertado": doc}), 200

# One single tomato analysis
@api_bp.route('/analizar_uno', methods=['POST'])
def analizar_uno():
    if 'imagen' not in request.files:
        return jsonify({"error": "Falta imagen"}), 400

    modelos = current_app.config["MODELOS"]
    segmentador = current_app.config["SEGMENTADOR"]
    CLASES = current_app.config["CLASES"]

    nombre_modelo = request.form.get('modelo', 'mobilenet').lower()
    modelo_actual = modelos.get(nombre_modelo)

    if not modelo_actual:
        return jsonify({"error": f"Modelo '{nombre_modelo}' no disponible"}), 400

    try:
        img_pil = Image.open(request.files['imagen']).convert('RGB')
        img_np = np.array(img_pil)
    except Exception:
        return jsonify({"error": "Imagen inválida"}), 400

    # 1. Segmentate with YOLO
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

    # 2. Classify
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
        "confianza": float(probs[idx]),
        "segmentacion": mensaje,
        "detalle": {k: float(v) for k, v in zip(CLASES, probs)}
    }), 200


# Multiple Segmentation
@api_bp.route('/segmentar_todo', methods=['POST'])
def segmentar_todo():
    if 'imagen' not in request.files:
        return jsonify({"error": "Falta imagen"}), 400

    modelos = current_app.config["MODELOS"]
    segmentador = current_app.config["SEGMENTADOR"]
    CLASES = current_app.config["CLASES"]

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
                except Exception:
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

    return send_file(img_io, mimetype='image/jpeg')
