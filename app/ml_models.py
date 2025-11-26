import os
import tensorflow as tf
from ultralytics import YOLO
import joblib
from inference_sdk import InferenceHTTPClient 

CLASES = ['dañado', 'maduro', 'verde', 'viejo']

def cargar_modelos(base_dir):
    modelos = {}
    segmentador = None
    roboflow_client = None

    try:
        yolo_path = os.path.join(base_dir, "models", "best.pt")
        if os.path.exists(yolo_path):
            segmentador = YOLO(yolo_path)
            print("✅ Segmentador YOLO Local cargado.")
        else:
            print("⚠️ No se encontró 'best.pt' local. Se dependerá de Roboflow.")
    except Exception as e:
        print(f"❌ Error cargando YOLO Local: {e}")

    api_key = os.getenv("ROBOFLOW_API_KEY") 
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

    try:
        mobilenet_path = os.path.join(base_dir, "models", "modelo_2_mobilenet_final.keras")
      
        if not os.path.exists(mobilenet_path):
             mobilenet_path = os.path.join(base_dir, "models", "modelo_2_mobilenet.keras")
        
        modelos['mobilenet'] = tf.keras.models.load_model(mobilenet_path)
        print(f"✅ Modelo Clasificación cargado desde: {os.path.basename(mobilenet_path)}")
    except Exception as e:
        print(f"❌ Error cargando MobileNet: {e}")

    try:
        svm_path = os.path.join(base_dir, "models", "modelo_3_svm.pkl")
        if os.path.exists(svm_path):
            modelos['svm'] = joblib.load(svm_path)
            print("✅ Modelo SVM cargado.")
    except Exception:
        pass

    return modelos, segmentador, roboflow_client, CLASES