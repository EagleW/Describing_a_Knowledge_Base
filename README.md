# Describing a Knowledge Base 

<b>[Describing a Knowledge Base](http://aclweb.org/anthology/W18-6502)</b>

Accepted by 11th International Conference on Natural Language Generation (INLG 2018)

 [[Slides]](https://eaglew.github.io/files/Wikipedia.pdf)

Table of Contents
=================
  * [Model Overview](#model-overview)
  * [Requirements](#requirements)
  * [Quickstart](#quickstart)
  * [Citation](#citation)
  * [Attention Visualization](#attention-visualization)

## Model Overview
<p align="center">
  <img src="https://eaglew.github.io/images/narratingkb.png?raw=true" alt="Photo" style="width: 100%;"/>
</p>

## Requirements
#### Environment:

- [Pytorch 0.4](http://pytorch.org/)
-  Python 3.6 **CAUTION!! Model might not be saved and loaded properly under Python 3.5**

#### Data: 

- [Wikipedia Person and Animal Dataset](https://drive.google.com/open?id=1TzcNdjZ0EsLh_rC1pBC7dU70jINcsVJd)<br>
This dataset gathers unfiltered 428,748 person and 12,236 animal infobox with description based on Wikipedia dump (2018/04/01) and Wikidata (2018/04/12)

## Quickstart
#### Preprocessing:
Put the [Wikipedia Person and Animal Dataset](https://drive.google.com/open?id=1TzcNdjZ0EsLh_rC1pBC7dU70jINcsVJd) under the `Describing a Knowledge Base` folder. Unzip it.

Randomly split the data into train, dev and test by runing `split.py` under utils folder.

```
python split.py
```

Run `preprocess.py` under the same folder. 

You can choose person (type 0) or animal (type 1)
```
python preprocess.py --type 0
```
#### Training
Hyperparameter can be adjusted in the Config class of main.py and choose whether person (0) or animal (1) using type.
```
python main.py --cuda --mode 0 --type 0
```

#### Test
Compute score:
```
python main.py --cuda --mode 3
```
Predict single entity:
```
python main.py --cuda --mode 1
```

## Citation

```
@InProceedings{W18-6502,
  author = 	"Wang, Qingyun
		and Pan, Xiaoman
		and Huang, Lifu
		and Zhang, Boliang
		and Jiang, Zhiying
		and Ji, Heng
		and Knight, Kevin",
  title = 	"Describing a Knowledge Base",
  booktitle = 	"Proceedings of the 11th International Conference on Natural Language Generation",
  year = 	"2018",
  publisher = 	"Association for Computational Linguistics",
  pages = 	"10--21",
  location = 	"Tilburg University, The Netherlands",
  url = 	"http://aclweb.org/anthology/W18-6502"
}
```
## Attention Visualization
<p align="center">
   <img src="https://eaglew.github.io/images/nkbtype.png?raw=true" alt="Photo" style="width:100%">
</p>
<p align="center">
    <img src="https://eaglew.github.io/images/nkbfinal.png?raw=true" alt="Photo" style="width:100%">
</p>
