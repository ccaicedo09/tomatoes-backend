import os
import tensorflow as tf
from ultralytics import YOLO
import joblib

CLASES = ['dañado', 'maduro', 'verde', 'viejo']

def cargar_modelos(base_dir):
    # A. Segmenter model
    try:
        yolo_path = os.path.join(base_dir, "models", "best.pt")
        segmentador = YOLO(yolo_path)
        print("✅ Segmentador YOLO cargado.")
    except Exception as e:
        print(f"❌ Error cargando YOLO: {e}")
        segmentador = None

    # B. Classification models
    modelos = {}

    try:
        cnn_path = os.path.join(base_dir, "models", "modelo_1_cnn.h5")
        modelos['cnn'] = tf.keras.models.load_model(cnn_path, compile=False)
    except Exception:
        print("⚠️ Aviso: No se encontró 'modelo_1_cnn.h5'.")

    try:
        mobilenet_final_path = os.path.join(base_dir, "models", "modelo_2_mobilenet_final.keras")
        modelos['mobilenet'] = tf.keras.models.load_model(mobilenet_final_path)
        print("✅ Modelo MobileNet final cargado.")
    except Exception as e:
        print(f"⚠️ Error con 'modelo_2_mobilenet_final.keras': {e}")
        try:
            mobilenet_old_path = os.path.join(base_dir, "models", "modelo_2_mobilenet.keras")
            modelos['mobilenet'] = tf.keras.models.load_model(mobilenet_old_path)
            print("✅ Modelo MobileNet (versión anterior) cargado.")
        except Exception as e2:
            print(f"❌ No se pudo cargar ningún modelo MobileNet: {e2}")

    try:
        svm_path = os.path.join(base_dir, "models", "modelo_3_svm.pkl")
        print("Comprobando existencia SVM:", os.path.exists(svm_path), "->", svm_path)
        modelos['svm'] = joblib.load(svm_path)
        print("✅ Modelo SVM cargado.")
    except Exception as e:
        print(f"❌ Error cargando modelo SVM: {e}")

    return modelos, segmentador, CLASES