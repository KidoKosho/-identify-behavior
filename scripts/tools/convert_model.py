"""
Tool script: Chuyển đổi mô hình Keras .h5 sang TensorFlow SavedModel.
"""
import os
import tensorflow as tf
import keras

keras.config.enable_unsafe_deserialization()

MODEL_H5_PATH = os.path.join("models", "fire_smoke", "model.h5")
EXPORT_DIR = os.path.join("models", "fire_smoke", "saved_model")

if os.path.exists(MODEL_H5_PATH):
    print(f"[*] Đang tải mô hình từ: {MODEL_H5_PATH} ...")
    model = tf.keras.models.load_model(MODEL_H5_PATH, compile=False)
    print(f"[*] Đang xuất sang format SavedModel: {EXPORT_DIR} ...")
    model.export(EXPORT_DIR)
    print(f"[✓] Đã xuất thành công tại: {EXPORT_DIR}")
else:
    print(f"[!] Không tìm thấy file model: {MODEL_H5_PATH}")
