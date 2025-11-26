import os
import tensorflow as tf
from ultralytics import YOLO
import joblib
from inference_sdk import InferenceHTTPClient # <--- NUEVO IMPORT

CLASES = ['dañado', 'maduro', 'verde', 'viejo']

def cargar_modelos(base_dir):
    modelos = {}
    segmentador = None
    roboflow_client = None

    # --- 1. Cargar YOLO Local (Plan A) ---
    try:
        yolo_path = os.path.join(base_dir, "models", "best.pt")
        if os.path.exists(yolo_path):
            segmentador = YOLO(yolo_path)
            print("✅ Segmentador YOLO Local cargado.")
        else:
            print("⚠️ No se encontró 'best.pt' local. Se dependerá de Roboflow.")
    except Exception as e:
        print(f"❌ Error cargando YOLO Local: {e}")

    # --- 2. Configurar Cliente Roboflow (Plan B - Potente) ---
    api_key = os.getenv("ROBOFLOW_API_KEY") # Lo tomaremos del .env
    if api_key:
        try:
            roboflow_client = InferenceHTTPClient(
                api_url="https://detect.roboflow.com",
                api_key=api_key
            )
            print("✅ Cliente Roboflow configurado.")
        except Exception as e:
            print(f"❌ Error configurando Roboflow: {e}")
    else:
        print("⚠️ No hay ROBOFLOW_API_KEY en .env")

    # --- 3. Cargar Modelos de Clasificación (MobileNet, etc) ---
    try:
        mobilenet_path = os.path.join(base_dir, "models", "modelo_2_mobilenet_final.keras")
        # Si tienes la versión vieja .h5, ajusta el nombre aquí
        if not os.path.exists(mobilenet_path):
             mobilenet_path = os.path.join(base_dir, "models", "modelo_2_mobilenet.keras")
        
        modelos['mobilenet'] = tf.keras.models.load_model(mobilenet_path)
        print(f"✅ Modelo Clasificación cargado desde: {os.path.basename(mobilenet_path)}")
    except Exception as e:
        print(f"❌ Error cargando MobileNet: {e}")

    # Cargar SVM si existe
    try:
        svm_path = os.path.join(base_dir, "models", "modelo_3_svm.pkl")
        if os.path.exists(svm_path):
            modelos['svm'] = joblib.load(svm_path)
            print("✅ Modelo SVM cargado.")
    except Exception:
        pass

    # Retornamos todo, incluyendo el cliente de Roboflow
    return modelos, segmentador, roboflow_client, CLASES