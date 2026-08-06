"""
YOLO Video Inference
Testa un modello YOLO su un video e mostra i risultati frame per frame.

Requisiti:
    pip install ultralytics opencv-python
"""

import cv2
from ultralytics import YOLO


def run_inference(
    model_path: str,
    video_path: str,
    output_path: str = None,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    show: bool = False,
    device: str = None,
):
    # --- Carica il modello ---
    print(f"[INFO] Caricamento modello: {model_path}")
    model = YOLO(model_path)
    if device:
        model.to(device)

    # --- Apri il video ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Impossibile aprire il video: {video_path}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] Video: {width}x{height} @ {fps:.1f} fps — {total} frame totali")

    # --- Writer per il video di output ---
    writer = None
    if output_path:
        # Nota: output deve essere un .mp4, non un .txt
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"[INFO] Salvataggio risultati in: {output_path}")

    frame_idx = 0
    print("[INFO] Avvio inferenza — premi 'q' per uscire, 's' per salvare uno screenshot\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # --- Inferenza ---
        results = model(frame, conf=conf_threshold, iou=iou_threshold, verbose=False)

        # --- Disegna le bounding box sul frame ---
        annotated = results[0].plot()

        # --- Statistiche a console ogni 30 frame ---
        if frame_idx % 30 == 0 or frame_idx == 1:
            det = results[0].boxes
            n   = len(det) if det is not None else 0
            pct = frame_idx / total * 100 if total > 0 else 0
            print(f"  Frame {frame_idx:>5}/{total} ({pct:5.1f}%)  —  {n} oggetti rilevati")

        # --- Overlay info ---
        cv2.putText(
            annotated,
            f"Frame: {frame_idx}  |  Conf >= {conf_threshold}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        # --- Mostra ---
        if show:
            cv2.imshow("YOLO Inference", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("\n[INFO] Uscita richiesta dall'utente.")
                break
            if key == ord("s"):
                screenshot = f"screenshot_frame_{frame_idx}.jpg"
                cv2.imwrite(screenshot, annotated)
                print(f"[INFO] Screenshot salvato: {screenshot}")

        if writer:
            writer.write(annotated)

    # --- Pulizia ---
    cap.release()
    if writer:
        writer.release()

    if show:
        cv2.destroyAllWindows()

    print(f"\n[DONE] Elaborati {frame_idx} frame.")
    if output_path:
        print(f"[DONE] Video salvato in: {output_path}")


if __name__ == "__main__":
    run_inference(
        model_path  = r"C:\Users\divic\Desktop\FVAB\esperimento_4\weights\best.pt",
        video_path  = r"C:\Users\divic\Desktop\FVAB\video_test.mp4",
        output_path = r"C:\Users\divic\Desktop\FVAB\output_esperimento_4.mp4",
        conf_threshold = 0.25,
        iou_threshold  = 0.45,
        show           = False,
        device         = None,
    )