# PointNet

## What this is about
Just a simple implementation based on the PointTransformer which was where people looked at PointNet and asked "can we parallelise the operations using Transformers?". Does pretty well but much of the same techniques in PointNet++ can be seen here. 

## What has been done 

1. Set up the Architecture - See /src/model.py
1. Set up the dataset and dataloader - See /src/data_preprocessing folder
1. Set up the training - see /src/train.py
1. Set up validation - Outputs classification reports
1. Results visualisation - check out ./notebooks/segmentation_inference.ipynb

## What needs to be done

1. All the above, but with Segmentation part. 

## Dataset Used

### Classification - ModelNet

The famous ModelNet dataset. The classification network was trained on ModelNet10.

### Part Segmentation - ShapeNet

The segmentation model was trained on the Airplane class in ShapeNet. Do note that the annotation is pretty good, and the parts annotated for one class is not the same as the next class. For example, the Airplane class has parts 0 - 3. The next class would have parts 4 and 5. So there is no overlap.  

## How to run 

Make sure you change the directory of your data.

```
python -m src.train
```

## Useful Sources

1. [Paper itself](https://arxiv.org/abs/2012.09164)
1. [A reference implementation](https://github.com/qq456cvb/Point-Transformers/tree/master)