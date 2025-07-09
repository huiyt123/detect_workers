import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import io
import cv2
import tempfile
import os
import warnings
from moviepy.video.io.VideoFileClip import VideoFileClip
warnings.filterwarnings("ignore")

def convert_avi_to_mp4(input_path, output_path):
    """使用moviepy转换AVI格式视频为MP4格式"""
    try:
        video = VideoFileClip(input_path)
        video.write_videofile(output_path, codec='libx264', audio_codec='aac')
        video.close()
        return True
        
    except Exception as e:
        print(f"转换失败: {str(e)}")
        return False

st.title("工人施工安全检测")

best_pt_path = "./runs/detect/train2/weights/best.pt"
model = YOLO(best_pt_path)

uploaded_file = st.file_uploader("上传图片或视频", type=["jpg", "jpeg", "png", "mp4", "avi", "mov", "mkv"], key="file_uploader")

if uploaded_file:
    file_type = uploaded_file.type
    if file_type.startswith("image"):
        # 处理图片
        image_bytes = uploaded_file.read()
        image = Image.open(io.BytesIO(image_bytes))
        image_np = np.array(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        st.subheader("上传的图像")
        st.image(image_np, caption="原始图像")
        with st.spinner("正在检测..."):
            results = model.predict(
                source=image_bgr,
                imgsz=640,
                conf=0.3,
                iou=0.6,
                device='0',
                save=True,
                show_labels=True,
                show_conf=True,
                project='runs/detect',
                name='image_predict'
            )
        st.subheader("检测结果")
        result_image = results[0].plot()
        result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        st.image(result_image, caption="检测结果")
        if len(results[0].boxes) > 0:
            st.subheader("检测统计")
            class_counts = {}
            for box in results[0].boxes:
                cls_id = int(box.cls.item())
                cls_name = model.names[cls_id]
                if cls_name in class_counts:
                    class_counts[cls_name] += 1
                else:
                    class_counts[cls_name] = 1
            for cls_name, count in class_counts.items():
                if cls_name == 'helmet':
                    st.write(f"检测到佩戴安全帽工人数量: {count}")
                elif cls_name == 'head':
                    st.write(f"检测到未佩戴安全帽工人数量: {count}")
    
    elif file_type.startswith("video"):
        # 保存上传的视频文件
        video_name = uploaded_file.name
        base_name = os.path.splitext(video_name)[0]
        with open(video_name, "wb") as f:
            f.write(uploaded_file.read())
        
        st.subheader("上传的视频")
        st.video(video_name)

        with st.spinner("正在处理视频..."):
            results = model.predict(
                source=video_name,
                imgsz=640,
                conf=0.3,
                iou=0.6,
                device='0',
                save=True,
                show_labels=True,
                show_conf=True,
                project='runs/detect',
                name='video_predict'
            )
        
        # 显示检测结果视频
        output_video_path = os.path.join(results[0].save_dir, base_name + '.avi')
        output_video_path_mp4 = output_video_path.replace('.avi', '.mp4')
        convert_avi_to_mp4(output_video_path, output_video_path_mp4)
        st.subheader("检测结果视频")
        st.video(output_video_path_mp4)
        
        # 清理临时文件
        if os.path.exists(video_name):
            os.remove(video_name)