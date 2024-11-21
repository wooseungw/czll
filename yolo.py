from ultralytics import YOLO

model = YOLO('yolo11x.pt')  # Load model
result = model.train(
    data='./data.yaml',
    name = "x_640_dropout025_",
    epochs=300,
    patience =10,
    batch=12,
    imgsz=640,
    optimizer = 'AdamW',
    lr0 = 0.001,
    
    dropout = 0.25,
)