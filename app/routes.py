import os
import uuid
from datetime import datetime

import io
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
from flask import Blueprint, current_app, request, jsonify, send_file

from .db import classification_samples, segmentation_samples

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
    
    save_sample = request.args.get("saveSample", "false").lower() == "true"

    modelos = current_app.config["MODELOS"]
    segmentador = current_app.config["SEGMENTADOR"]
    CLASES = current_app.config["CLASES"]

    nombre_modelo = request.form.get('modelo', 'mobilenet').lower()
    modelo_actual = modelos.get(nombre_modelo)

    if not modelo_actual:
        return jsonify({"error": f"Modelo '{nombre_modelo}' no disponible"}), 400

    file = request.files['imagen']
    original_filename = file.filename or "upload.jpg"

    try:
        img_pil = Image.open(request.files['imagen']).convert('RGB')
        img_np = np.array(img_pil)
    except Exception:
        return jsonify({"error": "Imagen inválida"}), 400
    
    height, width, channels = img_np.shape

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
        boxes = results[0].boxes
        boxes_np = boxes.xyxy.cpu().numpy().astype(int)

        # Detection info for Database samples
        num_detections = len(boxes_np)
        for i, b in enumerate(boxes_np):
            x1, y1, x2, y2 = [int(v) for v in b]
            # YOLO score and class from YOLO
            score = float(boxes.conf[i].cpu().numpy()) if boxes.conf is not None else None
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "class": "tomato",
                "score": score,
            })

        areas = [(box[2]-box[0]) * (box[3]-box[1]) for box in boxes_np]
        idx_mayor = int(np.argmax(areas))
        box = boxes_np[idx_mayor]

        h_img, w_img, _ = img_np.shape
        pad = 10
        x1 = max(0, box[0] - pad)
        y1 = max(0, box[1] - pad)
        x2 = min(w_img, box[2] + pad)
        y2 = min(h_img, box[3] + pad)

        recorte = img_np[y1:y2, x1:x2]
    else:
        mensaje = "⚠️ No se detectó tomate. Usando imagen completa."
        num_detections = 0
        detections = []

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
    
    predicted_label = CLASES[idx]
    predicted_confidence = float(probs[idx])
    predicted_probs = {k: float(v) for k, v in zip(CLASES, probs)}
    
    if save_sample:
        try:
            # Uploads folder + date
            date_str = datetime.utcnow().strftime("%Y%m%d")
            uploads_dir = os.path.join(current_app.root_path, "uploads", date_str)
            os.makedirs(uploads_dir, exist_ok=True)

            # Unique filename
            ext = os.path.splitext(original_filename)[1] or ".jpg"
            unique_name = f"{uuid.uuid4().hex}{ext}"
            full_path = os.path.join(uploads_dir, unique_name)

            # Save original image
            img_pil.save(full_path)

            # Relative route for storing on database
            rel_path = os.path.join("uploads", date_str, unique_name).replace(os.sep, "/")

            # Document for Mongo
            doc = {
                "filePath": rel_path,
                "originalFilename": original_filename,
                "uploadedAt": datetime.utcnow(),
                "width": width,
                "height": height,
                "channels": channels,

                "modelName": nombre_modelo,
                "modelVersion": None,  # versions not supported yet
                "predictedLabel": predicted_label,
                "predictedConfidence": predicted_confidence,
                "predictedProbs": predicted_probs,

                "numDetections": num_detections,
                "detections": detections,
                "annotatedFilePath": None,  # no annotated image present on method

                # Post-processing fields
                
                "postProcessing": {
                "trueLabel": None,
                "verified": False,
                "labeledBy": None,
                "labeledAt": None,
                "labelSource": None,
                "isCorrect": None,
                "errorType": None,
                }
            }

            classification_samples.insert_one(doc)
        except Exception as e:
            current_app.logger.error(f"Error guardando sample en DB: {e}")

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
    
    save_sample = request.args.get("saveSample", "false").lower() == "true"

    modelos = current_app.config["MODELOS"]
    segmentador = current_app.config["SEGMENTADOR"]
    CLASES = current_app.config["CLASES"]

    clasificador = modelos.get('mobilenet')
    if not segmentador or not clasificador:
        return jsonify({"error": "Faltan modelos (YOLO o MobileNet)"}), 500

    file = request.files['imagen']
    original_filename = file.filename or "upload.jpg"
    
    img_pil = Image.open(file).convert('RGB')
    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    img_original_np = np.array(img_pil)
    
    height, width, channels = img_original_np.shape

    results = segmentador.predict(
        img_original_np,
        conf=0.1,
        iou=0.7,
        retina_masks=True,
        verbose=False
    )
    result = results[0]
    
    num_detections = 0
    detections_for_db = []

    if result.boxes:
        
        boxes = result.boxes
        boxes_np = boxes.xyxy.cpu().numpy().astype(int)
        num_detections = len(boxes_np)
        
        for i, box in enumerate(boxes_np):
            x1, y1, x2, y2 = box

            if (x2 - x1) < 20 or (y2 - y1) < 20:
                continue

            recorte = img_original_np[y1:y2, x1:x2]
            if recorte.size == 0:
                continue

            recorte_pil = Image.fromarray(recorte).resize((224, 224))
            input_arr = tf.keras.preprocessing.image.img_to_array(recorte_pil)
            pred = clasificador.predict(np.array([input_arr]), verbose=0)
            probs = pred[0]
            idx = int(np.argmax(pred[0]))
            clase = CLASES[idx]
            conf = float(pred[0][idx])
            
            predicted_probs = {k: float(v) for k, v in zip(CLASES, probs)}

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
            (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img_bgr, (x1, y1 - 20), (x1 + w_text, y1), color, -1)
            cv2.putText(img_bgr, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 2)
            
            # DB Data
            area = int((x2 - x1) * (y2 - y1))
            score = float(boxes.conf[i].cpu().numpy()) if boxes.conf is not None else None

            detections_for_db.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "area": area,
                "class": "tomato",
                "score": score,                    # YOLO detection trust
                "predictedLabel": clase,           # segmentation classification
                "predictedConfidence": conf,
                "predictedProbs": predicted_probs
            })

    # If there are no result.boxes or all detections were filtered,
    # num_detections will remain 0 and detections_for_db will be empty [].
    # We still return the image without annotations.

    img_final_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_io = io.BytesIO()
    Image.fromarray(img_final_rgb).save(img_io, 'JPEG', quality=90)
    img_io.seek(0)
    
    if save_sample:
        try:
            # Make a dir using today's datr
            date_str = datetime.utcnow().strftime("%Y%m%d")

            # 1) Save original image
            uploads_dir = os.path.join(current_app.root_path, "uploads", date_str)
            os.makedirs(uploads_dir, exist_ok=True)

            ext = os.path.splitext(original_filename)[1] or ".jpg"
            original_name = f"{uuid.uuid4().hex}{ext}"
            original_full_path = os.path.join(uploads_dir, original_name)
            img_pil.save(original_full_path)

            original_rel_path = os.path.join("uploads", date_str, original_name).replace(os.sep, "/")

            # 2) Save segmented image
            outputs_dir = os.path.join(current_app.root_path, "outputs", date_str)
            os.makedirs(outputs_dir, exist_ok=True)

            annotated_name = f"{uuid.uuid4().hex}{ext}"
            annotated_full_path = os.path.join(outputs_dir, annotated_name)
            Image.fromarray(img_final_rgb).save(annotated_full_path, "JPEG", quality=90)

            annotated_rel_path = os.path.join("outputs", date_str, annotated_name).replace(os.sep, "/")

            # 3) Document for mongo
            doc = {
                "filePath": original_rel_path,
                "originalFilename": original_filename,
                "uploadedAt": datetime.utcnow(),
                "width": width,
                "height": height,
                "channels": channels,

                "modelName": "mobilenet",       # clasificador usado para cada recorte
                "modelVersion": None,
                "predictedLabel": None,         # not need bc it's segmentation
                "predictedConfidence": None,
                "predictedProbs": None,

                "numDetections": num_detections,
                "detections": detections_for_db,
                "annotatedFilePath": annotated_rel_path,

                # post processing labels for dataset fixing
                "postProcessing": {
                "trueLabel": None,
                "verified": False,
                "labeledBy": None,
                "labeledAt": None,
                "labelSource": None,
                "isCorrect": None,
                "errorType": None,
                }
            }
            
            segmentation_samples.insert_one(doc)

        except Exception as e:
            current_app.logger.error(f"Error guardando sample de segmentación en DB: {e}")

    return send_file(img_io, mimetype='image/jpeg')
