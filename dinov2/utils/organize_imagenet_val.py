import os
import json
import shutil

def load_imagenet_synsets(json_file):
    """加载 JSON 文件并返回 ILSVRC2012_ID 到 WNID 的映射"""
    with open(json_file, 'r') as f:
        data = json.load(f)

    id_to_wnid = {}
    for synset in data['data']['synsets']:
        ilsvrc_id = int(synset['ILSVRC2012_ID'][0])  # 注意这里是个列表
        wnid = synset['WNID']
        id_to_wnid[ilsvrc_id] = wnid

    return id_to_wnid

def load_ground_truth(gt_file):
    """加载 ground truth 文件，返回类别编号列表"""
    with open(gt_file, 'r') as f:
        lines = f.readlines()

    return [int(line.strip()) for line in lines]

def organize_images(val_dir, output_dir, label_map, id_to_wnid):
    val_images = sorted([img for img in os.listdir(val_dir) if img.endswith('.JPEG')])
    assert len(val_images) == len(label_map), "图像数量与标签数量不一致"

    for i, image_name in enumerate(val_images):
        class_id = label_map[i]
        wnid = id_to_wnid[class_id]
        target_dir = os.path.join(output_dir, wnid)
        os.makedirs(target_dir, exist_ok=True)
        src_path = os.path.join(val_dir, image_name)
        dst_path = os.path.join(target_dir, image_name)
        shutil.move(src_path, dst_path)  # 或者使用 copy

if __name__ == "__main__":
    # 设置路径
    json_file = '/root/autodl-tmp/imagenet/meta.json'              # 你的 JSON 文件
    val_dir = '/root/autodl-tmp/imagenet/val'                          # 原始 val 目录（含所有 JPEG）
    gt_file = '/root/autodl-tmp/imagenet/ILSVRC2012_validation_ground_truth.txt'   # 你的 ground truth 文件
    output_dir = val_dir                    # 输出目录，符合 DINOv2 格式

    # 步骤1: 加载 JSON 映射
    id_to_wnid = load_imagenet_synsets(json_file)

    # 步骤2: 加载 ground truth
    label_map = load_ground_truth(gt_file)

    # 步骤3: 整理图像
    organize_images(val_dir, output_dir, label_map, id_to_wnid)