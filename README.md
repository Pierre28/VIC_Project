# How do use

- Download dataset : [Kaggle](https://www.kaggle.com/drgilermo/face-images-with-marked-landmark-points#__sid=js0) et [iBug300W](https://ibug.doc.ic.ac.uk/download/300VW_Dataset_2015_12_14.zip/). Then, the directory should be organized as follows :

```
VIC_Project
|--- data
|      |--- Kaggle 
            |---- indices.txt
            |---- face_image...     
|      |--- 300W
|             |---- 01_Indoor
                    |---- indices.txt
                    |---- indoor_...
|             |---- 02_Outdoor   
                    |---- indices.txt
                    |---- outdoor...
|--- main.py
|--- model.py
|---  ..

```

To evaluate performances, run the following :

- Kaggle :

```
python main.py --dataset Kaggle
```

- iBug300W :

```
python main.py --dataset 300W
```
