import os
import shutil
import random
import xml.etree.ElementTree as ET
import yaml

def read_dataset_config(config_path='dataset.yaml'):
    if not os.path.exists(config_path):
        raise ValueError(f"Config file {config_path} does not exist.")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    if 'path' not in config:
        raise ValueError("Dataset path not found in the config file.")
    class_mapping = {name: idx for idx, name in config['names'].items()}

    # 兼容两个数据集的类别映射
    class_mapping['hat']=0
    class_mapping['person']=1
    return config['path'], class_mapping

def convert_xml_to_txt(xml_path, class_mapping):
    if not os.path.exists(xml_path):
        raise ValueError(f"XML file {xml_path} does not exist.")
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    img_width = int(size.find('width').text)
    img_height = int(size.find('height').text)
    yolo_lines = []
    
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        if class_name not in class_mapping:
            continue
            
        class_id = class_mapping[class_name]
        bndbox = obj.find('bndbox')
        
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)
        
        # 转换为 YOLO 格式（class_id, x_center, y_center, width, height）
        x_center = ((xmin + xmax) / 2) / img_width
        y_center = ((ymin + ymax) / 2) / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height
        
        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    return '\n'.join(yolo_lines)
    
def split_dataset(dataset_path, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, random_seed=0):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

    for subset in ['train', 'val', 'test']:
        os.makedirs(os.path.join(dataset_path, subset, 'images'), exist_ok=True)
        os.makedirs(os.path.join(dataset_path, subset, 'labels'), exist_ok=True)

    all_images = [
        f for f in os.listdir(os.path.join(dataset_path, 'images')) 
        if f.lower().endswith(('.jpg', '.png', '.jpeg'))
    ]
    
    if random_seed is not None:
        random.seed(random_seed)
    random.shuffle(all_images)

    # 将数据集划分为训练集、验证集和测试集
    total = len(all_images)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    subsets = {
        'train': all_images[:train_end],
        'val': all_images[train_end:val_end],
        'test': all_images[val_end:]
    }

    for subset, files in subsets.items():
        for img in files:
            src_img = os.path.join(dataset_path, 'images', img)
            dst_img = os.path.join(dataset_path, subset, 'images', img)
            shutil.copy(src_img, dst_img)
            
            base_name = os.path.splitext(img)[0]
            xml_path = os.path.join(dataset_path, 'annotations', base_name + '.xml')
            yolo_txt = convert_xml_to_txt(xml_path, class_mapping)
            txt_path = os.path.join(dataset_path, subset, 'labels', base_name + '.txt')
            with open(txt_path, 'w') as f:
                f.write(yolo_txt)
    
    print(f"Split completed: train={len(subsets['train'])}, "
          f"val={len(subsets['val'])}, test={len(subsets['test'])}")

if __name__ == "__main__":
    dataset_path, class_mapping = read_dataset_config()
    split_dataset(dataset_path)