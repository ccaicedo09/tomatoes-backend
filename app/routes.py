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
    
    
    rf_client = current_app.config.get("ROBOFLOW_CLIENT")
    modelos = current_app.config.get("MODELOS")
    clasificador = modelos.get('mobilenet') 
    CLASES = current_app.config.get("CLASES") 

    if not rf_client or not clasificador:
        return jsonify({"error": "Faltan modelos (Roboflow o MobileNet)"}), 500

    
    file = request.files['imagen']
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_original_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) 
    
    
    temp_filename = f"temp_{uuid.uuid4().hex}.jpg"
    cv2.imwrite(temp_filename, img_bgr)

    try:
        print("🚀 Enviando a Roboflow para segmentación...")
        result = rf_client.run_workflow(
            workspace_name="proyecto-kbmoz",
            workflow_id="detect-count-and-visualize-3",
            images={"image": temp_filename}
        )
        
        
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

        predictions = result[0].get("predictions", {}).get("predictions", [])
        print(f"✅ Roboflow detectó {len(predictions)} tomates.")

        
        for pred in predictions:
            
            x, y = int(pred["x"]), int(pred["y"])
            w, h = int(pred["width"]), int(pred["height"])
            
            
            x1 = max(0, int(x - w / 2))
            y1 = max(0, int(y - h / 2))
            x2 = min(img_bgr.shape[1], int(x + w / 2))
            y2 = min(img_bgr.shape[0], int(y + h / 2))

            
            recorte = img_original_rgb[y1:y2, x1:x2]
            
            label_final = "Desconocido"
            confianza_final = 0.0
            color = (255, 255, 255) 

            if recorte.size > 0:
                try:
                    
                    recorte_pil = Image.fromarray(recorte).resize((224, 224))
                    input_arr = tf.keras.preprocessing.image.img_to_array(recorte_pil)
                    input_arr = np.array([input_arr]) 

                    
                    preds = clasificador.predict(input_arr, verbose=0)
                    idx = int(np.argmax(preds[0]))
                    
                    label_final = CLASES[idx].upper() 
                    confianza_final = float(preds[0][idx])
                    
                    
                    colores = {
                        'DAÑADO': (0, 0, 255),    
                        'MADURO': (0, 165, 255),  
                        'VERDE': (0, 255, 0),     
                        'VIEJO': (128, 128, 128)  
                    }
                    color = colores.get(label_final, (255, 0, 0))
                    
                except Exception as e:
                    print(f"⚠️ Error clasificando recorte: {e}")

            
            if "points" in pred:
                pts = np.array([[int(p["x"]), int(p["y"])] for p in pred["points"]], np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(img_bgr, [pts], True, color, 2)

            texto = f"{label_final} {confianza_final*100:.0f}%"
            
            font_scale = 0.4  
            thickness = 1     
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            (text_w, text_h), _ = cv2.getTextSize(texto, font, font_scale, thickness)
            
            
            bg_h = text_h + 8
            cv2.rectangle(img_bgr, (x1, y1 - bg_h), (x1 + text_w + 4, y1), color, -1)
            
            
            cv2.putText(img_bgr, texto, (x1 + 2, y1 - 4), font, font_scale, (255, 255, 255), thickness)

    except Exception as e:
        print(f"❌ Error General: {e}")
        return jsonify({"error": str(e)}), 500

    img_final_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_io = io.BytesIO()
    Image.fromarray(img_final_rgb).save(img_io, 'JPEG', quality=95)
    img_io.seek(0)

    return send_file(img_io, mimetype='image/jpeg')