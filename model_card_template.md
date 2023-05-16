# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model predicts whether a person makes over 50K a year or less based on several explanatory factors affecting the income of a person like Age, Occupation, Education, etc. 
We have used multiple algorithms such as:

* Logistic Regression
* Extrat Classfier
* Support Vector Machines
* Gradient Boosting	
* Decision Trees
* Random Forest	
* XGB Classifier
* Naive Bayes
* K-Nearest Neighbor

We selected the xgboost algorithm with Baysian Optimization to optimized the hyperparameters in scikit-learn 1.2.2. The archivement was 85% accuracy on the testing data which can be upgrade with further experimentation.<br>

The optimal parameters that we found were:
* eta: `0.4`
* max_depth: `5`
* gamma: `0`

## Intended Use
This model is intended to be used to determine whether a person’s yearly income in US falls in the income category of either greater than 50K Dollars or less equal to 50K Dollars category based on a certain set of attributes.

## Training Data
We use the [Census Income](https://archive-beta.ics.uci.edu/dataset/20/census+income) Also known as [Adult dataset](https://archive-beta.ics.uci.edu/dataset/20/census+income).This dataset was wxtraction was done by Barry Becker from the 1994 Census database. The original dataset contains `32,561` rows and `15` columns. During ou cleansing process we have remove one of the columns which is the `education-num` since it was is highly overall correlated with `education`. We use `80%` of the data to train the model and `20%` to evaluate the model.
## Evaluation Data
we have used `20%` of the dataset to evaluate the model.

## Metrics
To evaluate the model we have used multiple metrics which are: `precision`, `recall`, `f1-score` and `accuracy`.

Test evaluation result:         
 * recall: `0.64` 
 * precision: `0.72`
 * f1-score: `0.68`
 * accuracy: `0.85`
 
Save the Confusion matrix: <br>
[[`4558`  `390`] <br>
 [`563`  `997`]]

## Ethical Considerations
Datasets are central to the machine learning ecosystem. In many cases, the $50k threshold understates and misrepresents the broader picture. The `Capital_loss` column also has majority of the values set as `0`, similar to `Capital_gains`. The hours per week column has values scattered over a range of `1–99` while majority of the values have data near 40 hours. This might limits the external validity of the prediction task.

## Caveats and Recommendations
Extraction was done from the 1994 Census database. The data is quite old, and the insights drawn cannot be directly used for derivation in the modern world and cannot adequately be used as a statistical representation of the population.

