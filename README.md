# MOTS_SiamMask
2019 Summer Research Project, aim at realizing Multi Object Tracking and Segmentation by combining MaskRCNN and SiamMask.

## 项目构成:

### SiamMask
Single object tracker  [Code](https://github.com/foolwood/SiamMask)



### MaskRCNN

Object detector with mask  [Code](https://github.com/matterport/Mask_RCNN)



### ReID

Reid model which is used to compute the distance between two objects  [Code](https://github.com/layumi/Person_reID_baseline_pytorch)



### Dataset

MOTS dataset, including MOTSChallenge and KITTY MOTS  [Project Page](https://www.vision.rwth-aachen.de/page/mots)



### mots_tools
MOTS visualization and evaluation tools



### Result
Folder to save tracking result



### main.py
Tracking code



## Reqiurement
See requirements.txt in MaskRCNN and SiamMask



## Usage

```python
python main.py
```



## Citation

MaskRCNN:

```
@ARTICLE{2017arXiv170306870H,
       author = {{He}, Kaiming and {Gkioxari}, Georgia and {Doll{\'a}r}, Piotr and
         {Girshick}, Ross},
        title = "{Mask R-CNN}",
      journal = {arXiv e-prints},
     keywords = {Computer Science - Computer Vision and Pattern Recognition},
         year = "2017",
        month = "Mar",
          eid = {arXiv:1703.06870},
        pages = {arXiv:1703.06870},
archivePrefix = {arXiv},
       eprint = {1703.06870},
 primaryClass = {cs.CV},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2017arXiv170306870H},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

SiamMask:

```
@ARTICLE{2018arXiv181205050W,
       author = {{Wang}, Qiang and {Zhang}, Li and {Bertinetto}, Luca and {Hu}, Weiming and
         {Torr}, Philip H.~S.},
        title = "{Fast Online Object Tracking and Segmentation: A Unifying Approach}",
      journal = {arXiv e-prints},
     keywords = {Computer Science - Computer Vision and Pattern Recognition},
         year = "2018",
        month = "Dec",
          eid = {arXiv:1812.05050},
        pages = {arXiv:1812.05050},
archivePrefix = {arXiv},
       eprint = {1812.05050},
 primaryClass = {cs.CV},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2018arXiv181205050W},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

ReID:

```
@article{DBLP:journals/corr/SunZDW17,
  author    = {Yifan Sun and
               Liang Zheng and
               Weijian Deng and
               Shengjin Wang},
  title     = {SVDNet for Pedestrian Retrieval},
  booktitle   = {ICCV},
  year      = {2017},
}

@article{hermans2017defense,
  title={In Defense of the Triplet Loss for Person Re-Identification},
  author={Hermans, Alexander and Beyer, Lucas and Leibe, Bastian},
  journal={arXiv preprint arXiv:1703.07737},
  year={2017}
}
```

