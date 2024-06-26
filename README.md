# PointTransformer

## What this is about
Just a simple implementation based on the PointTransformer which was where people looked at PointNet and asked "can we parallelise the operations using Transformers?". Does pretty well but much of the same techniques in PointNet++ can be seen here. 

Will cover PTv1 - v3. 

## What has been done 

1. Set up the Architecture - See /src/model.py
1. Set up the dataset and dataloader - See /src/data_preprocessing folder
1. Set up the training - see /src/train.py
1. Set up validation - Outputs classification reports
1. Results visualisation - check out ./notebooks/segmentation_inference.ipynb
1. All the above, but with Segmentation part. 

## What needs to be done

1. PTV3 setup

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

## Results from Training

### PointTransformerV1

Classification Results:

| Classification Report: |   Loss | Accuracy |         |     |
|-----------------------:|-------:|---------:|---------|-----|
|              precision | recall | f1-score | support |     |
|                      0 |   1.00 |     0.02 | 0.04    | 50  |
| 1                      | 0.79   | 0.91     | 0.85    | 100 |
| 2                      | 0.81   | 0.97     | 0.88    | 100 |
| 3                      | 0.68   | 0.27     | 0.39    | 85  |
| 4                      | 0.57   | 0.80     | 0.67    | 86  |
| 5                      | 0.91   | 0.90     | 0.90    | 100 |
| 6                      | 0.70   | 0.41     | 0.52    | 85  |
| 7                      | 0.59   | 0.96     | 0.73    | 100 |
| 8                      | 0.76   | 0.87     | 0.81    | 100 |
| 9                      | 0.85   | 0.75     | 0.80    | 100 |
| accuracy               |        |          | 0.73    | 906 |
| macro avg              | 0.77   | 0.69     | 0.66    | 906 |
| weighted avg           | 0.76   | 0.73     | 0.70    | 906 |

Segmentation Results:

| Classification Report: |           |        |          |         |
|-----------------------:|----------:|-------:|----------|---------|
|                        | precision | recall | f1-score | support |
|                      0 |      0.85 |   0.96 | 0.90     | 432108  |
|                      1 | 0.93      | 0.86   | 0.89     | 294364  |
|                      2 | 0.95      | 0.71   | 0.81     | 122813  |
|                      3 | 0.85      | 0.86   | 0.85     | 110715  |
| accuracy               | 0.89      |        |          | 960000  |
| macro avg              | 0.90      | 0.85   | 0.87     | 960000  |
| weighted avg           | 0.89      | 0.89   | 0.88     | 960000  |

### PTV2 results:

Classification Results:

| Classification Report: |           |        |          |         |
|-----------------------:|----------:|-------:|----------|---------|
|                        | precision | recall | f1-score | support |
|                      0 |      1.00 |   0.80 | 0.89     | 50      |
|                      1 | 0.95      | 0.97   | 0.96     | 100     |
|                      2 | 0.89      | 0.98   | 0.93     | 100     |
|                      3 | 0.76      | 0.71   | 0.73     | 86      |
|                      4 | 0.80      | 0.92   | 0.85     | 86      |
|                      5 | 1.00      | 0.99   | 0.99     | 99      |
|                      6 | 0.80      | 0.70   | 0.75     | 86      |
|                      7 | 0.94      | 0.99   | 0.97     | 100     |
|                      8 | 0.85      | 0.82   | 0.83     | 100     |
|                      9 | 0.97      | 0.98   | 0.97     | 99      |
|               accuracy | 0.90      |        |          | 906     |
|              macro avg | 0.90      | 0.89   | 0.89     | 906     |
|           weighted avg | 0.90      | 0.90   | 0.89     | 906     |

Segmentation Results (20 Epochs):

| Classification Report: |           |        |          |         |
|-----------------------:|----------:|-------:|----------|---------|
|                        | precision | recall | f1-score | support |
|                      0 |      0.78 |   0.98 | 0.87     | 432074  |
|                      1 | 0.97      | 0.81   | 0.88     | 294411  |
|                      2 | 0.99      | 0.33   | 0.50     | 122817  |
|                      3 | 0.77      | 0.92   | 0.84     | 110698  |
|               accuracy | 0.84      |        |          | 960000  |
|              macro avg | 0.88      | 0.76   | 0.77     | 960000  |
|           weighted avg | 0.86      | 0.84   | 0.82     | 960000  |



## Useful Sources

1. [Paper itself](https://arxiv.org/abs/2012.09164)
1. [A reference implementation](https://github.com/qq456cvb/Point-Transformers/tree/master)