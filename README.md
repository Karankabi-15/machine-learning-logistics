Logistic regression is a statistical method used in machine learning for binary classification problems, where the outcome or dependent variable can take only two possible types: 0 or 1, true or false, positive or negative, etc. Despite its name, logistic regression is actually a classification algorithm rather than a regression algorithm.

### Key Concepts of Logistic Regression

**Binary Classification:**
- Logistic regression is used to predict the probability of a binary outcome based on one or more predictor variables (features).
- The output is a probability value between 0 and 1, which can be thresholded to predict the binary class.

**Logistic Function (Sigmoid Function):**
- The logistic function, also known as the sigmoid function, maps any real-valued number into the range [0, 1]:
  \[
  \sigma(z) = \frac{1}{1 + e^{-z}}
  \]
  where \( z \) is the linear combination of input features and their corresponding coefficients:
  \[
  z = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \ldots + \beta_p x_p
  \]
  Here:
  - \( \beta_0 \) is the intercept.
  - \( \beta_1, \beta_2, \ldots, \beta_p \) are the coefficients of the predictor variables.
  - \( x_1, x_2, \ldots, x_p \) are the predictor variables.

**Probability Prediction:**
- The logistic regression model predicts the probability \( P(Y=1|X) \) as:
  \[
  P(Y=1|X) = \sigma(\beta_0 + \beta_1 x_1 + \beta_2 x_2 + \ldots + \beta_p x_p)
  \]
- The predicted class is obtained by applying a threshold (usually 0.5) to the predicted probability.

### Model Fitting

**Maximum Likelihood Estimation (MLE):**
- The coefficients of the logistic regression model are estimated using maximum likelihood estimation, which finds the values that maximize the likelihood of observing the given data.
- The likelihood function for logistic regression is:
  \[
  L(\beta) = \prod_{i=1}^{n} P(Y_i|X_i)^{Y_i} (1 - P(Y_i|X_i))^{1 - Y_i}
  \]
  where \( P(Y_i|X_i) \) is the predicted probability for the \( i \)-th observation.

**Log-Likelihood:**
- It is common to work with the log-likelihood, which is easier to optimize:
  \[
  \log L(\beta) = \sum_{i=1}^{n} \left( Y_i \log(P(Y_i|X_i)) + (1 - Y_i) \log(1 - P(Y_i|X_i)) \right)
  \]

### Assumptions of Logistic Regression
1. **Binary Outcome:** The dependent variable is binary.
2. **Linearity:** There is a linear relationship between the log-odds of the outcome and the predictor variables.
3. **Independence:** Observations are independent of each other.
4. **No Multicollinearity:** Predictor variables are not highly correlated with each other.

### Evaluation Metrics
- **Accuracy:** The proportion of correctly classified instances.
- **Precision:** The proportion of true positive predictions among all positive predictions.
- **Recall (Sensitivity):** The proportion of true positive predictions among all actual positives.
- **F1 Score:** The harmonic mean of precision and recall.
- **ROC Curve and AUC:** The Receiver Operating Characteristic (ROC) curve plots the true positive rate against the false positive rate, and the Area Under the Curve (AUC) measures the overall performance of the model.

### Applications of Logistic Regression
- **Medical Diagnosis:** Predicting the presence or absence of a disease based on patient features.
- **Spam Detection:** Classifying emails as spam or not spam.
- **Credit Scoring:** Assessing the risk of loan default based on financial history and other factors.
- **Marketing:** Predicting whether a customer will purchase a product based on demographic and behavioral data.

### Advantages and Disadvantages

**Advantages:**
- Simple and easy to implement.
- Provides probabilities and classifies.
- Computationally efficient and works well with small datasets.
- Can handle non-linear effects by including interaction terms or using non-linear transformations of the predictors.

**Disadvantages:**
- Assumes a linear relationship between the log-odds of the outcome and predictor variables.
- Not suitable for non-linear relationships unless the model is extended.
- Can struggle with multicollinearity and irrelevant variables.

Logistic regression is a fundamental tool in binary classification tasks, offering simplicity, interpretability, and a solid foundation for understanding more complex models.
