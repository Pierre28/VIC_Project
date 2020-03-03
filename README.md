# How do use

- Download dataset : [Kaggle](https://www.kaggle.com/drgilermo/face-images-with-marked-landmark-points#__sid=js0) et [iBug300W](https://ibug.doc.ic.ac.uk/download/300VW_Dataset_2015_12_14.zip/). Then, the directory should be organized as follows :

```
VIC_Project
|--- data
|      |--- Kaggle      
|      |--- 300W
|             |---- 01_Indoor
|             |---- 02_Outdoor   
|--- main.py
|--- model.py
|---  ..

```

# Presentation

- Intro
- Slide Pourquoi ? A quoi ça sert. Quelles application downstream
- Slide overview (timeline comment on attaque). ASM -> AAM -> DPM 
- Dig in : ASM (formules)
- Dig in ASM + SIFT
- Résultats



#  Useful info/links

- [Dataset website](https://ibug.doc.ic.ac.uk/resources/300-W/)
    - Note : evaluation pipeline and scripts provided
- [Mastering OpenCV projects](https://github.com/MasteringOpenCV)
- [AAM/ASM notebook](https://ibug.doc.ic.ac.uk/media/uploads/notebook.html)


# Todo

Now:
@high
- Fix Mahalanobis distance
- Add constrainsts on shape model
- Investigate low self.deformable_model values

@medium
- Implement pose estimation
- Investigate image instance re-sampling from constraint shape model

Future : 
- Read SIFT-ASM
- Read / Implement AAM
- Read DMP


/!\ 
- Model b is very small at the moment. And eigenvalue might not make sense. 
- Matching model to new points is executed in 1 step (because currently no scale/rot applied). Make sure it doesn't when used with those

/!\
ASM : Mahablabla distance not very good: covariance too small : exploding values. Fix this under dataset_utils : find a good strategy / distance / k s.t gs close to g -> min dist

# Ideas/Notes

- 300-W images are not of same size + not centered around face. 
- Restrict to front face ?
- Additional dataset to consider :
    - [KAGGLE](https://www.kaggle.com/drgilermo/face-images-with-marked-landmark-points#__sid=js0)
    - [List of datasets](https://www.ecse.rpi.edu/~cvrl/database/Facial_Landmark_Databases.html)
    - [3D face ](http://mhug.disi.unitn.it/workshop/3dfaw/)


# Paper summary

## [300 Faces in-the-Wild Challenge](./papers/Sagonas_300_Faces_in-the-Wild_2013_ICCV_paper.pdf)

- Dataset caracteristics :
    - 300 *indoors*, 300 *outdoors*, collected from google. 
    - Frequent occlusion.
    - 68 occlusion points.
    - Face bounding boxes retreive using face detection algorithm

- Baseline Method : Appearance Model algorithm ([Link](https://www.ri.cmu.edu/pub_files/pub4/matthews_iain_2003_1/matthews_iain_2003_1.pdf)), using edge-feature structures defined in ([Link](https://ieeexplore.ieee.org/document/990655)).

- Evaluation : Pointwise Euclidean distance between landmark and prediction (For both 68-points and 51 inner points)

- 6 participants to the challenge :
    - Probabilistic pactc expert + Optimisation.
    - Active Shape Models including modified SIFT descriptors [Link](http://www.milbo.org/stasm-files/multiview-active-shape-models-with-sift-for-300w.pdf)
    - Conv-network
    - Local Evidence Aggregated Regression + Support Vector Regressors. [Link](https://ieeexplore.ieee.org/document/6755921)
    - Nearest-Neightbours using global descriptor + alignment using locally linear model. Energy based minimzation to combine both. [Link](http://openaccess.thecvf.com/content_iccv_workshops_2013/W11/papers/Hasan_Localizing_Facial_Keypoints_2013_ICCV_paper.pdf)

Q. Understand Pt-Pt evaluation results Plots


## [Multiview ASM]((http://www.milbo.org/stasm-files/multiview-active-shape-models-with-sift-for-300w.pdf))

- [ASM-SIFT for template matching](http://www.milbo.org/stasm-files/active-shape-models-with-sift-and-mars.pdf)(same authors)


# Active shape model

- ASM is closely related to AAM (Active appearance model)

- [ASM - Original paper](http://www.tamaraberg.com/teaching/Fall_13/papers/asm.pdf)
- [Better paper](https://pdfs.semanticscholar.org/ebc2/ceba03a0f561dd2ab27c97b641c649c48a14.pdf)

Summary : 

1. Face images are challenging, since objects have inherent variability of shape, pose, making it difficult to fit naive schemes. Model-based approach attempt to fit a prior model to the current sample. Thus, this is a *top-bottom* strategy, different from *bottom-up* strategy that consider local regions, searching for edges or corner and reconstructing a global structure from them. Here, a global model exists.
2. Many types of prior model exist, each suited for a particular problem. 
3. Interesting landmark are landmark located at high curvature or junction. 
4. The shape of an object is nor- mally considered to be independent of the position, orientation and scale of that object. Alignment is required before training. (See paper appendix).
5. (4.1.3) Possible to apply PCA on the landmark to reduce model dimension. And generate new landmark by modifying landmark in the PC space, and projecting back, as long as modification are bounding by eigenvalue.
6. (4.1.4 Fitting model to new points. (Iterative minimisation. Pure geometry).
7. **Matching model instance to an image**. (4.2.1) Take a model c. Project it to the image (PCA + RT ou juste PCA?), and compute a fit function F(c). Chosen model defined by c_ is c_ = argming F(c). *Thus, in theory all we have to do is to choose a suitable fit function, and use a general purpose optimiser to find the minimum*. Un peu bêbête un effet.
8. (4.2 - ) Multiple possibility in choosing the fit function. (Minimal distance between a model point and strong edges (sensitive to prior edge detector), similarity in model point neighbourhoods compared to thoses observed in traning set).
9. (4.2 - ) The requires an optimization strategy to find the best set of parameters (b, Xt, Yt, theta, s). In case no prior information is known on where to find the image, Genetic Algorithm can be used. When some knowledge, use simplex. Otherwise can derive a ASM specific method.(Active shape model)
9. Sample along profile normal of model points on the training set, build a statistical model of this point. Sample k pixels, put in vector g_i (sample the derivative actually. Not sure if they mean sampling sobel, or derivate the vector itself) + normalize. The set {g_i} models a given point. Assume this is a multivariate gaussian, with mean g_ and covariance Sg. Then use Mahalanobis distance to attest the quality of a fit. Minimizing f(gs) is equivalent to maximizing the probability that gs comes from this distribution. 
During training, sample a profile of m > k pixels, and create this gs vector for 2(m-k) + 1 and measure the fit function. The best position is chosen as new point xi. Iterating over all possible points, we get a new instance of the model in the image X'. 
Use algorithm 4.1.4 to compute new shape, pose/scale parameters for the new instance X'.

Note : how to choose the number of modes (pca dimension ?)
- Explained variance criterion
- Reach enough accuracy to any face in the dataset. 

Q. Is b a fuking vector or a matrix? 
**STOP 4.2.2**

-> Model is basically linear combination of landmarks scheme seen during training, projected using PCA. Fiding suitable model means finding the one minimizing a handcrafted specific fit function, that, in essence, will make sure landmark are located at edges or corners.

# Deformable part model (DPM)

- [8000+citations](http://cs.brown.edu/people/pfelzens/papers/lsvm-pami.pdf)


# SIFT descriptors

- [SIFT descriptor](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf). 55,000 citations.. 

**Scale Invariant Feature Transform**

- Features, invariant to image scaling, rotation/
- Partially invariant to change in illumination, 3D camera viewpoint.
- Using spatial and frequential domains.
- Highly distrinctive features (can match easily a given a given feature in a list of features)

Pipeline of feature extraction :

1. Scale-space extrema detection : Using difference-of-Gaussian (DoG), search over all scale and location potential interest points.
    - DoG used to remove high-frequency features (textures, ..)
2. Keypoint location: A model is used to score each candidate, and determine location and scale.
3. Orientation assignement: orientation is assigned to each keypoint (how?).
4. Keypoint descriptor : gradient computed around each descriptor at the selected scale. 

TO Complete : 3 Detection of scale-space extrema

# To Read :

- [300-W annotation procedure](https://ibug.doc.ic.ac.uk/media/uploads/documents/sagonas_cvpr_2013_amfg_w.pdf)
- [300W results](../papers/sagonas_2016_imavis.pdf)
- [Active shape model](http://www.tamaraberg.com/teaching/Fall_13/papers/asm.pdf)
    - Note: seems to have 1D gradients vs 2D gradients


