# Model Card for Census Income Classifier

## Model Details

This is a logistic regression model which aims to classify whether a person has an income
greater than or equal to $50,000 based on various demographc features. The model is trained
on data from the UCI census income data set (https://archive.ics.uci.edu/ml/datasets/census+income).

## Intended Use

This model has been built as part of an MLOPS project and not meant to be used for production.

## Training Data

The training data consists of a sample composed of 80% of the census data, stratified based on
salary. All categorical values have been one-hot encoded. There are no null values in the data,
however there are some features that use "?" to denote an unknown value. "?" has been
treated as a legitimate value and the model has been trained with data containing "?"'s

## Evaluation Data

The evaluation data consists of a sample composed of 20% of the census data, stratified based
on salary, with no overlap with the training data.

## Metrics

The model has been evaluated based on 3 metrics - precision, recall and fbeta. For the overall model, these metrics are as follows

- Precision: 0.73
- Recall: 0.61
- fbeta: 0.66

Slice performance of the model is as follows

### Slicing category: **workclass**

| Categorical Value | Precision | Recall | fbeta |
| ----------------- | --------- | ------ | ----- |
|  State-gov                    |   0.77 |   0.67 |   0.71 |
|  Self-emp-not-inc             |   0.71 |   0.48 |   0.58 |
|  Private                      |   0.74 |   0.58 |   0.65 |
|  Federal-gov                  |   0.73 |   0.73 |   0.73 |
|  Local-gov                    |    0.7 |   0.63 |   0.66 |
|  ?                            |   0.67 |   0.38 |   0.49 |
|  Self-emp-inc                 |   0.77 |   0.82 |    0.8 |
|  Without-pay                  |      1 |      1 |      1 |
|  Never-worked                 |      1 |      1 |      1 |

### Slicing category: **education**

| Categorical Value | Precision | Recall | fbeta |
| ----------------- | --------- | ------ | ----- |
|  Bachelors                    |   0.74 |    0.8 |   0.77 |
|  HS-grad                      |   0.68 |   0.27 |   0.38 |
|  11th                         |      1 |   0.13 |   0.24 |
|  Masters                      |    0.8 |   0.87 |   0.83 |
|  9th                          |    0.5 |  0.037 |  0.069 |
|  Some-college                 |   0.65 |   0.46 |   0.54 |
|  Assoc-acdm                   |   0.67 |   0.52 |   0.59 |
|  Assoc-voc                    |   0.69 |   0.48 |   0.56 |
|  7th-8th                      |    0.5 |   0.05 |  0.091 |
|  Doctorate                    |   0.83 |   0.89 |   0.86 |
|  Prof-school                  |   0.84 |   0.92 |   0.88 |
|  5th-6th                      |      1 |   0.12 |   0.22 |
|  10th                         |    0.5 |  0.097 |   0.16 |
|  1st-4th                      |      1 |   0.17 |   0.29 |
|  Preschool                    |      0 |      1 |      0 |
|  12th                         |   0.82 |   0.27 |   0.41 |

### Slicing category: **marital-status**

| Categorical Value | Precision | Recall | fbeta |
| ----------------- | --------- | ------ | ----- |
|  Never-married                |   0.79 |   0.27 |    0.4 |
|  Married-civ-spouse           |   0.73 |   0.65 |   0.69 |
|  Divorced                     |   0.81 |   0.32 |   0.45 |
|  Married-spouse-absent        |   0.58 |   0.41 |   0.48 |
|  Separated                    |    0.8 |   0.36 |    0.5 |
|  Married-AF-spouse            |      1 |    0.4 |   0.57 |
|  Widowed                      |   0.76 |   0.33 |   0.46 |

### Slicing category: **occupation**

| Categorical Value | Precision | Recall | fbeta |
| ----------------- | --------- | ------ | ----- |
|  Adm-clerical                 |   0.66 |   0.42 |   0.51 |
|  Exec-managerial              |   0.78 |    0.8 |   0.79 |
|  Handlers-cleaners            |   0.71 |   0.14 |   0.23 |
|  Prof-specialty               |   0.76 |    0.8 |   0.78 |
|  Other-service                |    0.7 |    0.1 |   0.18 |
|  Sales                        |   0.71 |    0.6 |   0.65 |
|  Craft-repair                 |   0.65 |    0.3 |   0.41 |
|  Transport-moving             |   0.66 |   0.28 |    0.4 |
|  Farming-fishing              |   0.75 |   0.26 |   0.39 |
|  Machine-op-inspct            |   0.69 |   0.16 |   0.27 |
|  Tech-support                 |    0.7 |   0.67 |   0.68 |
|  ?                            |   0.67 |   0.38 |   0.49 |
|  Protective-serv              |   0.64 |   0.46 |   0.54 |
|  Armed-Forces                 |      1 |      1 |      1 |
|  Priv-house-serv              |   0.33 |      1 |    0.5 |

### Slicing category: **relationship**

| Categorical Value | Precision | Recall | fbeta |
| ----------------- | --------- | ------ | ----- |
|  Not-in-family                |   0.77 |   0.32 |   0.46 |
|  Husband                      |   0.74 |   0.65 |   0.69 |
|  Wife                         |    0.7 |    0.7 |    0.7 |
|  Own-child                    |   0.69 |   0.13 |   0.23 |
|  Unmarried                    |   0.87 |   0.25 |   0.39 |
|  Other-relative               |   0.61 |    0.3 |    0.4 |

### Slicing category: **race**

| Categorical Value | Precision | Recall | fbeta |
| ----------------- | --------- | ------ | ----- |
|  White                        |   0.74 |   0.61 |   0.67 |
|  Black                        |   0.73 |    0.5 |   0.59 |
|  Asian-Pac-Islander           |   0.71 |   0.59 |   0.65 |
|  Amer-Indian-Eskimo           |   0.59 |   0.56 |   0.57 |
|  Other                        |   0.48 |   0.44 |   0.46 |

### Slicing category: **sex**

| Categorical Value | Precision | Recall | fbeta |
| ----------------- | --------- | ------ | ----- |
|  Male                         |   0.74 |   0.61 |   0.67 |
|  Female                       |   0.72 |   0.53 |   0.61 |

### Slicing category: **native-country**

| Categorical Value | Precision | Recall | fbeta |
| ----------------- | --------- | ------ | ----- |
|  United-States                |   0.74 |    0.6 |   0.66 |
|  Cuba                         |   0.83 |    0.6 |    0.7 |
|  Jamaica                      |   0.33 |    0.2 |   0.25 |
|  India                        |   0.72 |   0.78 |   0.75 |
|  ?                            |   0.72 |    0.6 |   0.66 |
|  Mexico                       |   0.67 |   0.18 |   0.29 |
|  South                        |   0.47 |   0.56 |   0.51 |
|  Puerto-Rico                  |    0.7 |   0.58 |   0.64 |
|  Honduras                     |      1 |      1 |      1 |
|  England                      |   0.69 |   0.67 |   0.68 |
|  Canada                       |   0.69 |   0.56 |   0.62 |
|  Germany                      |   0.76 |   0.57 |   0.65 |
|  Iran                         |   0.74 |   0.78 |   0.76 |
|  Philippines                  |   0.79 |   0.54 |   0.64 |
|  Italy                        |    0.8 |   0.48 |    0.6 |
|  Poland                       |   0.44 |   0.33 |   0.38 |
|  Columbia                     |    0.4 |      1 |   0.57 |
|  Cambodia                     |      1 |   0.14 |   0.25 |
|  Thailand                     |   0.75 |      1 |   0.86 |
|  Ecuador                      |    0.5 |    0.5 |    0.5 |
|  Laos                         |      1 |    0.5 |   0.67 |
|  Taiwan                       |   0.85 |   0.85 |   0.85 |
|  Haiti                        |   0.67 |    0.5 |   0.57 |
|  Portugal                     |      1 |   0.25 |    0.4 |
|  Dominican-Republic           |   0.33 |    0.5 |    0.4 |
|  El-Salvador                  |   0.73 |   0.89 |    0.8 |
|  France                       |    0.7 |   0.58 |   0.64 |
|  Guatemala                    |      1 |   0.33 |    0.5 |
|  China                        |    0.6 |   0.75 |   0.67 |
|  Japan                        |   0.76 |   0.67 |   0.71 |
|  Yugoslavia                   |    0.6 |    0.5 |   0.55 |
|  Peru                         |      1 |    0.5 |   0.67 |
|  Outlying-US(Guam-USVI-etc)   |      1 |      1 |      1 |
|  Scotland                     |      1 |   0.67 |    0.8 |
|  Trinadad&Tobago              |      1 |    0.5 |   0.67 |
|  Greece                       |    0.6 |   0.75 |   0.67 |
|  Nicaragua                    |      0 |      0 |      0 |
|  Vietnam                      |   0.25 |    0.2 |   0.22 |
|  Hong                         |   0.83 |   0.83 |   0.83 |
|  Ireland                      |      1 |    0.6 |   0.75 |
|  Hungary                      |      1 |   0.33 |    0.5 |
|  Holand-Netherlands           |      1 |      1 |      1 |

## Ethical Considerations

There may be factors other than those considered here which may affect income. It is quite
possible that the salary distribution in the data sample may not be representative of the
salary distribution in the population, particularly with regard to race and sex. Additional
research and data augmentation may be required before this model can be put to use.

## Caveats and Recommendations

This model should not be used to make financial decisions given the ethical considerations
described in the prior section. It should be used simply as an academic exercise. Note that
the sample size is fairly small and may not accurately reflect the salary distributions in
the population. The dataset is clearly class imbalanced as well as country imbalanced.

## References

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf
