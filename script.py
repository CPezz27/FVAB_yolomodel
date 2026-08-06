import os
import time
import cv2
import numpy as np
import rospy
from queue import Queue, Empty
from sensor_msgs.msg import Image

# ==========================================
# 1. AUTO-CONFIGURAZIONE E IMPORT GPU
# ==========================================
os.environ["ROS_MASTER_URI"] = "http://localhost:11311"
os.environ["ROS_HOSTNAME"] = "localhost"
if "DISPLAY" not in os.environ:
    os.environ["DISPLAY"] = ":0"

# Importiamo le librerie a basso livello NVIDIA
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit # Inizializza automaticamente il contesto CUDA

# ==========================================
# 2. COSTANTI E VARIABILI GLOBALI
# ==========================================
MODEL_PATH = "best320.engine"
INPUT_W = 320
INPUT_H = 320
CONF_THRESH = 0.5 # Soglia di confidenza (ignoriamo predizioni sotto il 50%)
IOU_THRESH = 0.45 # Soglia per la Non-Maximum Suppression (evita box doppi)

MY_CLASS_NAMES = {
    0:  "stop_signal",
    1:  "limit10",
    2:  "limit20",
    3:  "forward_right",
    4:  "green_light",
    5:  "yellow_light",
    6:  "red_light",
    7:  "pedon",
    8:  "front_car",
    9:  "bottom_car",
    10: "left_car",
    11: "right_car",
}

# ==========================================
# 3. IL MOTORE TENSORRT (WRAPPER C++)
# ==========================================
class YOLOv8_TRT:
    """Classe per gestire direttamente l'allocazione di memoria sulla GPU."""
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self.logger, namespace="")
        
        # Deserializziamo il file .engine
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        
        # Allocazione della memoria (Host = RAM, Device = VRAM GPU)
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem, 'shape': self.engine.get_binding_shape(binding)})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem, 'shape': self.engine.get_binding_shape(binding)})

    def infer(self, input_image):
        """Copia l'immagine sulla GPU, esegue il calcolo e riporta i risultati sulla RAM."""
        # Copia i dati contigui nell'area di memoria bloccata della CPU
        np.copyto(self.inputs[0]['host'], input_image.ravel())
        
        # Trasferimento Asincrono CPU -> GPU
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        
        # Esecuzione matematica sui Tensor Core
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        # Trasferimento Asincrono GPU -> CPU
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()
        
        return self.outputs[0]['host'], self.outputs[0]['shape']

# ==========================================
# 4. GESTIONE VIDEO E LOGICA NMS
# ==========================================
frame_queue = Queue(maxsize=1)

def callback(msg):
    frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if frame_queue.full():
        try: frame_queue.get_nowait()
        except Empty: pass
    frame_queue.put(frame)

def pulisci_coda():
    while not frame_queue.empty():
        try: frame_queue.get_nowait()
        except Empty: break

def preprocess_image(frame):
    """Prepara l'immagine per la rete neurale senza usare Ultralytics."""
    img = cv2.resize(frame, (INPUT_W, INPUT_H))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normalizzazione da 0-255 a 0.0-1.0 (fondamentale per la precisione FP16)
    img = img.astype(np.float32) / 255.0
    
    # HWC (Altezza, Larghezza, Colore) -> CHW (Colore, Altezza, Larghezza)
    img = np.transpose(img, (2, 0, 1))
    
    # Assicuriamoci che la memoria sia contigua in C prima di passarla a CUDA
    img = np.ascontiguousarray(img)
    return img

def postprocess_and_draw(frame, output_data, output_shape):
    """Decodifica la matrice generata da TensorRT e disegna i box (Ottimizzato)."""
    out = output_data.reshape(output_shape)
    preds = out[0].T  # Forma: (8400, 16)
    
    x_scale = frame.shape[1] / INPUT_W
    y_scale = frame.shape[0] / INPUT_H

    # 1. Separiamo la matrice: i primi 4 valori sono i box, il resto sono le classi
    boxes_raw = preds[:, :4]
    classes_raw = preds[:, 4:]
    
    # 2. Troviamo la classe con la confidenza massima per ogni riga in blocco
    scores = np.max(classes_raw, axis=1)
    class_ids = np.argmax(classes_raw, axis=1)
    
    # 3. Creiamo una maschera booleana per filtrare SUBITO i box sotto la soglia
    mask = scores > CONF_THRESH
    
    # 4. Applichiamo la maschera per tenere solo le rilevazioni buone (pochissime rispetto a 8400)
    filtered_boxes = boxes_raw[mask]
    filtered_scores = scores[mask]
    filtered_class_ids = class_ids[mask]
    
    boxes_list = []
    scores_list = []
    class_ids_list = []
    
    if len(filtered_boxes) > 0:
        # 5. Convertiamo le coordinate in blocco
        cx = filtered_boxes[:, 0]
        cy = filtered_boxes[:, 1]
        w = filtered_boxes[:, 2]
        h = filtered_boxes[:, 3]
        
        left = ((cx - w / 2) * x_scale).astype(np.int32)
        top = ((cy - h / 2) * y_scale).astype(np.int32)
        width = (w * x_scale).astype(np.int32)
        height = (h * y_scale).astype(np.int32)
        
        # 6. Ricreiamo la lista nel formato richiesto da OpenCV per l'NMS
        boxes_list = np.column_stack((left, top, width, height)).tolist()
        scores_list = filtered_scores.tolist()
        class_ids_list = filtered_class_ids.tolist()

    # NMS: Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(boxes_list, scores_list, CONF_THRESH, IOU_THRESH)
    
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes_list[i]
            label = f"{MY_CLASS_NAMES[class_ids_list[i]]} {scores_list[i]:.2f}"
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
    return frame

# ==========================================
# 5. MODALITA' YOLO
# ==========================================
trt_engine = None

def run_yolo():
    global trt_engine
    print("\n[INFO] Avvio YOLO Object Detection - Modalita' TensorRT Pura...")
    
    if trt_engine is None:
        print(f"[INFO] Inizializzazione memoria GPU per '{MODEL_PATH}'...")
        trt_engine = YOLOv8_TRT(MODEL_PATH)
        print("[INFO] Motore agganciato con successo ai Tensor Core!")

    pulisci_coda()
    camera_sub = rospy.Subscriber("/depth_cam/rgb/image_raw", Image, callback)
    
    prev_time = time.time()
    
    while not rospy.is_shutdown():
        try:
            frame = frame_queue.get(timeout=0.05)
            original_frame = frame.copy()
            
            # 1. Pre-processing CPU
            input_tensor = preprocess_image(frame)
            
            # 2. Inferenza Pura GPU (Tensor Core)
            output_data, output_shape = trt_engine.infer(input_tensor)
            
            # 3. Post-processing CPU e Disegno
            annotated_frame = postprocess_and_draw(original_frame, output_data, output_shape)
            
            # Calcolo FPS (Dovresti vederli molto più alti ora)
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time)
            prev_time = curr_time
            cv2.putText(annotated_frame, f"TRT FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            
            cv2.imshow("YOLO TensorRT - JetAuto", annotated_frame)

        except Empty:
            pass

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n[INFO] Chiusura finestra YOLO...")
            break

    camera_sub.unregister()
    cv2.destroyAllWindows()

# ==========================================
# 6. MENU MAIN (Semplificato per il test)
# ==========================================
def main():
    rospy.init_node("jetauto_trt_vision", anonymous=True)
    while not rospy.is_shutdown():
        print("\n--- JETAUTO TENSORRT VISION ---")
        print("1. Avvia Object Detection (Nativo GPU)")
        print("2. Esci")
        scelta = input("Scelta: ")

        if scelta == '1': run_yolo()
        elif scelta == '2': break

if __name__ == '__main__':
    try: main()
    except rospy.ROSInterruptException: pass