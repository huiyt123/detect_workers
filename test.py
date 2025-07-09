import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from ultralytics import YOLO
import argparse

def test_inference(model, path):
    # path可以是图片、视频或包含多张图片的文件夹的路径
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} does not exist.")
    model(
        path,
        imgsz=640,
        conf=0.3,
        iou=0.6,
        device='0',
        save=True,
        show_labels=False,
        show_conf=False,
        )

def test_dataset_validation(model, dataset_yaml_path):
    model.val(
        data=dataset_yaml_path,
        split='test',
        imgsz=640,
        conf=0.3,
        iou=0.6,
        plots=True,
        verbose=True,
        device='0'
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['infer', 'val'], required=True, help='Choose infer (inference) or val (validation)')
    parser.add_argument('--path', type=str, help='Path to image/video/folder for inference, required in infer mode')
    parser.add_argument('--weights', type=str, default="./runs/detect/train5/weights/best.pt", help='Path to model weights')
    parser.add_argument('--data', type=str, default="dataset.yaml", help='Dataset config yaml, only needed in val mode')
    args = parser.parse_args()

    model = YOLO(args.weights)

    if args.mode == 'infer':
        if not args.path:
            raise ValueError("You must provide a path for inference mode.")
        if not os.path.exists(args.path):
            raise FileNotFoundError(f"The specified path {args.path} does not exist.")
        test_inference(model, args.path)
    elif args.mode == 'val':
        test_dataset_validation(model, args.data)

if __name__ == "__main__":
    main()
