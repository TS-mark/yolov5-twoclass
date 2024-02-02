# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
在检测数据集上对训练好的YOLOv5检测模型进行验证

用法：
    $ python val.py --weights yolov5s.pt --data coco128.yaml --img 640

用法 - 格式：
    $ python val.py --weights yolov5s.pt                 # PyTorch
                              yolov5s.torchscript        # TorchScript
                              yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                              yolov5s_openvino_model     # OpenVINO
                              yolov5s.engine             # TensorRT
                              yolov5s.mlmodel            # CoreML (macOS-only)
                              yolov5s_saved_model        # TensorFlow SavedModel
                              yolov5s.pb                 # TensorFlow GraphDef
                              yolov5s.tflite             # TensorFlow Lite
                              yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                              yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import json
import os
import sys
from pathlib import Path
import shutil

import numpy as np
import torch
from tqdm import tqdm

# 确定YOLOv5根目录并将其添加到sys.path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5根目录
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # 添加ROOT到PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # 相对路径

# 导入所需的模块和函数
from models.common import DetectMultiBackend
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.general import (LOGGER, TQDM_BAR_FORMAT, Profile, check_dataset, check_img_size, check_requirements,
                           check_yaml, coco80_to_coco91_class, colorstr, increment_path, non_max_suppression,
                           print_args, scale_boxes, xywh2xyxy, xyxy2xywh)
from utils.metrics import ConfusionMatrix, ap_per_class, box_iou
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.torch_utils import select_device, smart_inference_mode


def save_one_txt(predn, save_conf, shape, file):
    # 保存一个txt结果
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # 归一化增益whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # 归一化xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # 标签格式
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')


def save_one_json(predn, jdict, path, class_map):
    # 保存一个JSON结果 {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy中心转换为左上角坐标
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append({
            'image_id': image_id,
            'category_id': class_map[int(p[5])],
            'bbox': [round(x, 3) for x in b],
            'score': round(p[4], 5)})


def process_batch(detections, labels, iouv):
    """
    返回正确预测的矩阵
    参数:
        detections (array[N, 6])，x1、y1、x2、y2、conf、class
        labels (array[M, 5])，class、x1、y1、x2、y2
    返回:
        correct (array[N, 10])，对于10个IoU水平
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    # print("a.device{}   b.device{}".format((labels[:,1:].device),detections[:,:4].device))
    iou = box_iou(labels[:, 1:], detections[:, :4])

    correct_class = labels[:, 0:1] == detections[:, 5]

    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > 阈值且类别匹配
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)


@smart_inference_mode()
def run(
        data,
        weights=None,  # model.pt路径
        batch_size=32,  # 批处理大小
        imgsz=640,  # 推理尺寸（像素）
        conf_thres=0.001,  # 置信度阈值
        iou_thres=0.6,  # NMS的IoU阈值
        max_det=300,  # 每张图像的最大检测数量
        task='val',  # 训练、验证、测试、速度或学习任务
        device='',  # cuda设备，如0或0,1,2,3或cpu
        workers=8,  # 最大数据加载器工作线程数（在DDP模式下每个RANK）
        single_cls=False,  # 将数据集视为单类别数据集
        augment=False,  # 增强推理
        verbose=True,  # 详细输出
        save_txt=False,  # 将结果保存为*.txt文件
        save_hybrid=False,  # 将标签+预测混合结果保存为*.txt文件
        save_conf=False,  # 在--save-txt标签中保存置信度
        save_json=False,  # 保存为COCO-JSON结果文件
        project=ROOT / 'runs/val',  # 保存到项目/名称
        name='exp',  # 保存到项目/名称
        exist_ok=False,  # 允许存在的项目/名称，不增加后缀
        half=True,  # 使用FP16半精度推理
        dnn=False,  # 使用OpenCV DNN进行ONNX推理
        model=None,
        dataloader=None,
        save_dir=Path(''),
        plots=True,
        callbacks=Callbacks(),
        compute_loss=None,
):

    # 标签映射
    label_mapping = {}
    import yaml
    # print(data)
    with open(data, "r") as f:
        a = yaml.safe_load(f)
        for i in range(a["nc"]):
            label_mapping[i] = [a["lb1"][i], a["lb2"][i]]



    # 初始化/加载模型并设置设备
    training = model is not None
    if training:  # 被train.py调用
        device, pt, jit, engine = next(model.parameters()).device, True, False, False  # 获取模型设备，PyTorch模型
        half &= device.type != 'cpu'  # 仅在CUDA上支持半精度
        model.half() if half else model.float()
    else:  # 直接调用
        device = select_device(device, batch_size=batch_size)

        # 目录
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # 增加运行次数
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # 创建目录

        # 加载模型
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
        imgsz = check_img_size(imgsz, s=stride)  # 检查图像尺寸
        half = model.fp16  # 有限的后端支持FP16
        if engine:
            batch_size = model.batch_size
        else:
            device = model.device
            if not (pt or jit):
                batch_size = 1  # export.py模型默认使用批处理大小为1
                LOGGER.info(f'对于非PyTorch模型，强制使用--batch-size 1的方形推理（1,3,{imgsz},{imgsz}）')

        # 数据
        data = check_dataset(data)  # 检查数据集


    # 配置
    model.eval()
    cuda = device.type != 'cpu'
    is_coco = isinstance(data.get('val'), str) and data['val'].endswith(f'coco{os.sep}val2017.txt')  # COCO数据集
    nc = 1 if single_cls else int(data['nc'])  # 类别数量
    # nc = 2
    iouv = torch.linspace(0.3, 0.75, 10, device=device)  # mAP@0.5:0.95的IoU向量
    niou = iouv.numel()

    # 数据加载器
    if not training:
        if pt and not single_cls:  # 检查--weights是否训练于--data上
            ncm = model.model.nc
            # assert ncm == nc, f'{weights} ({ncm} classes)训练于与您传递的--data不同的数据上（{nc}个类别）。请传递正确的--weights和--data组合，它们应该是一起训练的。'
        model.warmup(imgsz=(1 if pt else batch_size, 3, imgsz, imgsz))  # 预热
        pad, rect = (0.0, False) if task == 'speed' else (0.5, pt)  # 基准测试的方形推理
        pad = 0.0
        task = task if task in ('train', 'val', 'test') else 'val'  # 训练/验证/测试图像的路径
        dataloader = create_dataloader(data[task],
                                       imgsz,
                                       batch_size,
                                       stride,
                                       single_cls,
                                       pad=pad,
                                       rect=rect,
                                       workers=workers,
                                       prefix=colorstr(f'{task}: '))[0]

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = model.names if hasattr(model, 'names') else model.module.names  # 获取类别名称
    if isinstance(names, (list, tuple)):  # 旧格式
        names = dict(enumerate(names))
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    s = ('%22s' + '%11s' * 6) % ('Class', 'Images', 'Instances', 'P', 'R', 'mAP50', 'mAP50-95')
    tp, fp, p, r, f1, mp, mr, map50, ap50, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    dt = Profile(), Profile(), Profile()  # 分析时间
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    callbacks.run('on_val_start')
    pbar = tqdm(dataloader, desc=s, bar_format=TQDM_BAR_FORMAT)  # 进度条
    nc1 = data['nc1']
    nc2 = data['nc2']


    # 实际训练中有些类别被融合起来了，这里归为一类
    val_label_fusion = True
    two_cls_mapping = {
         0: 0,
         1: 0,
         2: 2,
         3: 2,
         4: 4,
         5: 4,
         6: 6,
         7: 6,
         8: 8,
         9: 8,
         10: 10,
         11: 10,
         12: 12,
         13: 12,
         14: 14,
         15: 14,
         16: 16,
         17: 16,
         18: 18,
         19: 18,
         20: 20,
         21: 20,
         22: 22,
         23: 22,
         24: 24,
         25: 25,
         26: 26,
         27: 27,
         28: 28,
         29: 29,
         30: 30,
         31: 31,
         32: 25,
         33: 26,
         34: 27,
         35: 35,
         36: 36,
         37: 35,
         38: 35,
         39: 39,
         40: 40,
         41: 41,
         42: 42,
         43: 43,
         44: 44,
         45: 45,
         46: 26,
         47: 25,
         48: 26,
         49: 25
    }

    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
        callbacks.run('on_val_batch_start')
        with dt[0]:
            if cuda:
                im = im.to(device, non_blocking=True)
                targets = targets.to(device)
            im = im.half() if half else im.float()  # 将图像数据类型转换为半精度浮点数（fp16）或单精度浮点数（fp32）
            im /= 255  # 将像素值从 0-255 缩放到 0.0-1.0 范围
            nb, _, height, width = im.shape  # 获取批次大小、通道数、图像高度和宽度

        # 推断
        with dt[1]:
            preds, train_out = model(im) if compute_loss else (model(im, augment=augment), None)

        if isinstance(preds, list):
            preds = preds[0]
        # TLD map
        mapped_preds = torch.zeros((preds.shape[0], preds.shape[1], nc + 5))
        mapped_preds[:, :, 0:5] = preds[:, :, 0:5]
        """这里可以用张量计算优化一下"""
        for j in range(nc):
            mapped_preds[:, :, 5+j] = (preds[:, :, 5 + label_mapping[j][0]] + preds[:, :, 5 + nc1 +label_mapping[j][1]]) / 2
        mapped_preds = mapped_preds.to(device)
        preds = mapped_preds

        # 计算损失
        if compute_loss:
            loss += compute_loss(train_out, targets)[1][:3] # 损失函数（box, obj, cls）

        # NMS（非极大值抑制）
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # 将目标框坐标转换为像素坐标
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # 用于自动标注的目标框
        with dt[2]:
            preds = non_max_suppression(preds,
                                        conf_thres,
                                        iou_thres,
                                        labels=lb,
                                        multi_label=False,
                                        agnostic=single_cls,
                                        max_det=max_det)

        # 评估指标
        for si, pred in enumerate(preds):
            # print("si", si, "preds", preds)
            labels = targets[targets[:, 0] == si, 1:]
            nl, npr = labels.shape[0], pred.shape[0]  # 标签数，预测框数


            predn = pred.clone()
            predn = predn.to(device)

            if val_label_fusion:
            # tld_twocls
                predn[:, 5] = torch.tensor([two_cls_mapping.get(x.item(), x.item()) for x in predn[:, 5]])
                pred[:, 5] = torch.tensor([two_cls_mapping.get(x.item(), x.item()) for x in pred[:, 5]])
                labels[:, 0] = torch.tensor([two_cls_mapping.get(x.item(), x.item()) for x in labels[:, 0]])



            path, shape = Path(paths[si]), shapes[si][0]
            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # 初始化正确标记
            seen += 1
            if npr == 0:
                if nl:
                    stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
                    if plots:
                        confusion_matrix.process_batch(detections=None, labels=labels[:, 0])
                continue

            if single_cls:
                pred[:, 5] = 0

            # 预测值标签合并，目标值标签合并
            scale_boxes(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # 缩放预测框坐标到原始图像空间
            # 评估
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # 目标框坐标
                scale_boxes(im[si].shape[1:], tbox, shape, shapes[si][1])  # 缩放目标框坐标到原始图像空间
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # 目标框坐标（原始图像空间）


                correct = process_batch(predn, labelsn, iouv)
                if plots:
                    confusion_matrix.process_batch(predn, labelsn)

            stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # 记录统计数据（正确标记、置信度、预测类别、目标类别）

            # 保存/记录
            if save_txt:
                save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / f'{path.stem}.txt')  # 保存预测框到文本文件


            if save_json:
                save_one_json(predn, jdict, path, class_map)  # 追加到 COCO-JSON 字典
            callbacks.run('on_val_image_end', pred, predn, path, names, im[si])  # 回调函数

        # 绘制图像
        if plots and batch_i < 3:
            plot_images(im, targets, paths, save_dir / f'val_batch{batch_i}_labels.jpg', names)  # 绘制目标标签图像
            plot_images(im, output_to_target(preds), paths, save_dir / f'val_batch{batch_i}_pred.jpg',
                        names)  # 绘制预测结果图像

        callbacks.run('on_val_batch_end', batch_i, im, targets, paths, shapes, preds)  # 回调函数

        # 绘制图像
        if plots and batch_i < 3:
            plot_images(im, targets, paths, save_dir / f'val_batch{batch_i}_labels.jpg', names)  # 绘制目标标签图像
            plot_images(im, output_to_target(preds), paths, save_dir / f'val_batch{batch_i}_pred.jpg',
                        names)  # 绘制预测结果图像

        callbacks.run('on_val_batch_end', batch_i, im, targets, paths, shapes, preds)  # 回调函数


    # 计算指标
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # 转换为numpy数组
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    nt = np.bincount(stats[3].astype(int), minlength=nc)  # 每个类别的目标数量

    # 打印结果
    pf = '%22s' + '%11i' * 2 + '%11.3g' * 4  # 打印格式
    LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
    if nt.sum() == 0:
        LOGGER.warning(f'警告 ⚠️ {task}集中找不到标签，无法计算指标')

    # 按类别打印结果
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # 打印速度
    t = tuple(x.t / seen * 1E3 for x in dt)  # 每张图像的速度
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        LOGGER.info(f'速度: 预处理%.1fms，推理%.1fms，NMS%.1fms（每张图像，尺寸为{shape}）' % t)

    # 绘图
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        callbacks.run('on_val_end', nt, tp, fp, p, r, f1, ap, ap50, ap_class, confusion_matrix)

    # 保存JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # 权重文件名
        anno_json = str(Path(data.get('path', '../coco')) / 'annotations/instances_val2017.json')  # 标注的JSON文件路径
        pred_json = str(save_dir / f"{w}_predictions.json")  # 预测结果的JSON文件路径
        LOGGER.info(f'\n评估 pycocotools mAP... 保存为 {pred_json}...')
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            check_requirements('pycocotools')
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # 初始化标注API
            pred = anno.loadRes(pred_json)  # 初始化预测API
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.im_files]  # 要评估的图像ID
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # 更新结果（mAP@0.5:0.95, mAP@0.5）
        except Exception as e:
            LOGGER.info(f'pycocotools 无法运行: {e}')

    # 返回结果
    model.float()  # 用于训练
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} 个标签已保存到 {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"结果已保存至 {colorstr('bold', save_dir)}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / './data/TLD_train.yaml',
                        help='dataset.yaml路径')
    # '../datasets/TLD_train5/TLD_train.yaml'
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'runs/train/TLD_twocls9/weights/best.pt',
                        help='模型路径')
    parser.add_argument('--batch-size', type=int, default=64, help='批量大小')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='推理尺寸（像素）')
    parser.add_argument('--conf-thres', type=float, default=0.01, help='置信度阈值')
    # parser.add_argument('--conf-thres', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--iou-thres', type=float, default=0.25, help='NMS IoU阈值')
    parser.add_argument('--max-det', type=int, default=300, help='每张图像的最大检测数')
    parser.add_argument('--task', default='val', help='train, val, test, speed或study')
    parser.add_argument('--device', default='2', help='cuda设备，例如0或0,1,2,3或cpu')
    parser.add_argument('--workers', type=int, default=8, help='数据加载器的最大工作线程数（DDP模式中的每个RANK）')
    parser.add_argument('--single-cls', action='store_true', help='将数据集视为单类别数据集')
    parser.add_argument('--augment', action='store_true', help='增强推理')
    parser.add_argument('--verbose', action='store_true', help='按类别报告mAP')
    parser.add_argument('--save-txt', action='store_true', help='将结果保存为*.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='将标签+预测混合结果保存为*.txt')
    parser.add_argument('--save-conf', action='store_true', help='在--save-txt标签中保存置信度')
    parser.add_argument('--save-json', action='store_true', help='保存COCO-JSON格式的结果文件')
    parser.add_argument('--project', default=ROOT / 'runs/val', help='保存路径为project/name')
    parser.add_argument('--name', default='ggwp', help='保存路径为project/name')
    parser.add_argument('--exist-ok', action='store_true', help='允许使用现有的project/name路径，不递增')
    parser.add_argument('--half', action='store_true', help='使用FP16半精度推理')
    parser.add_argument('--dnn', action='store_true', help='使用OpenCV DNN进行ONNX推理')
    # parser.add_argument('--print-err-pic', action='store_true', help='输出错误的图像')
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # 检查YAML文件
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.save_txt |= opt.save_hybrid
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))

    if opt.task in ('train', 'val', 'test'):  # 正常运行
        if opt.conf_thres > 0.001:  # https://github.com/ultralytics/yolov5/issues/1466
            LOGGER.info(f'警告 ⚠️ 置信度阈值 {opt.conf_thres} > 0.001 会产生无效结果')
        if opt.save_hybrid:
            LOGGER.info('警告 ⚠️ --save-hybrid将返回来自混合标签的高mAP，而不是仅从预测中获取的mAP')
        run(**vars(opt))

    else:
        weights = opt.weights if isinstance(opt.weights, list) else [opt.weights]
        opt.half = torch.cuda.is_available() and opt.device != 'cpu'  # 使用FP16加速推理
        if opt.task == 'speed':  # 进行速度基准测试
            # python val.py --task speed --data coco.yaml --batch 1 --weights yolov5n.pt yolov5s.pt...
            opt.conf_thres, opt.iou_thres, opt.save_json = 0.25, 0.45, False
            for opt.weights in weights:
                run(**vars(opt), plots=False)

        elif opt.task == 'study':  # 进行速度与mAP的基准测试
            # python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5n.pt yolov5s.pt...
            for opt.weights in weights:
                f = f'study_{Path(opt.data).stem}_{Path(opt.weights).stem}.txt'  # 保存结果的文件名
                x, y = list(range(256, 1536 + 128, 128)), []  # x轴（图像尺寸），y轴
                for opt.imgsz in x:  # 图像尺寸
                    LOGGER.info(f'\nRunning {f} --imgsz {opt.imgsz}...')
                    r, _, t = run(**vars(opt), plots=False)
                    y.append(r + t)  # 结果和时间
                np.savetxt(f, y, fmt='%10.4g')  # 保存结果
            os.system('zip -r study.zip study_*.txt')
            plot_val_study(x=x)  # 绘制图表


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
