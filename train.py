import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import warnings
from ultralytics import YOLO
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    data_yaml_path = './dataset.yaml'
    pretrain_weights_path = './yolo11s.pt'
    model = YOLO(pretrain_weights_path)
    model.train(
        data=data_yaml_path,
        epochs=200,
        patience=50,
        optimizer='AdamW',
        lr0=0.001,
        warmup_epochs=5,
        imgsz=640,
        batch=16,
        device='0'
    )

    # # 继续训练
    # model = YOLO('./runs/detect/train2/weights/last.pt')
    # model.train(resume=True)

    
