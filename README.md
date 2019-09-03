# MOTS_SiamMask
2019悉尼大学暑期研究课题, 在MOTS数据集上, 利用单目标跟踪的工作 SiamMask 结合目标检测工作 Mask RCNN 实现多目标跟踪与分割任务

## 项目构成:

### SiamMask
单目标跟踪模块

### MaskRCNN
带mask的目标检测模块

### ReID
reid 模块, 用于计算 object 之间 appearance 的距离

### Dataset
MOTS 数据集, 包括 MOTSChallenge 和 KITTY MOTS

### mots_tools
MOTS 评估及可视化工具

### Result
用于Tracking 结果保存

### main.py
Tracking 主逻辑代码

## Reqiurement
See requirements.txt in MaskRCNN and SiamMask

## Usage
```python
python main.py
```