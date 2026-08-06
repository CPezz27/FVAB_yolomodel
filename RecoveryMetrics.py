from ultralytics import YOLO

model = YOLO(r"C:\Users\divic\runs\esperimento_0\weights\best.pt")

print("Calcolo delle metriche in corso...")
metrics = model.val()

precision = metrics.box.mp
recall = metrics.box.mr       
map50 = metrics.box.map50     
map50_95 = metrics.box.map

print("\n--- METRICHE ESTRATTE ---")
print(f"Precision:      {precision:.4f}")
print(f"Recall:         {recall:.4f}")
print(f"mAP@0.5:        {map50:.4f}")
print(f"mAP@0.5:0.95:   {map50_95:.4f}")