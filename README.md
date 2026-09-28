# Machine Learning Labs

A collection of hands-on Jupyter notebook labs introducing core machine-learning ideas through guided implementations, experiments, and written reports. The labs progress from Python and data handling to supervised learning, neural networks, and unsupervised clustering. Exercises ask to write and implement the code on own.

## Getting started

### Clone the repository

```bash
git clone https://github.com/Masters-Halmstad/Learning-System-Lab.git
cd Learning-System-Lab
```

### Install Python packages

The notebooks use Python 3 and the following packages: Jupyter Notebook, NumPy, SciPy, Matplotlib, scikit-learn, and `stemming` (used in the email-spam notebook).

On Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install notebook numpy scipy matplotlib scikit-learn stemming
```

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install notebook numpy scipy matplotlib scikit-learn stemming
```

### Open and run a lab

Start Jupyter from the lab's folder so that notebook-relative paths to its datasets, images, and helper files resolve correctly. For example:

```bash
cd "Lab 2 - Regression"
jupyter notebook
```

In the browser, open the notebook for the exercise you want to do. Run the cells in order, follow the notebook instructions, and complete its marked exercises. For another lab, stop Jupyter, change to that lab's folder, and start it again. Some labs use additional files in their own `datasets/`, `imgs/`, or helper-code subfolders; keep the repository's folder structure intact.

## Labs and key concepts

| Lab | Notebook(s) | Main concepts and practical data |
| --- | --- | --- |
| [Lab 1 - Basics](./Lab%201%20-%20Basic/Lab%201%20-%20Basics.ipynb) ([report](./Lab%201%20-%20Basic/ML_LAB_1_Report.pdf)) | Introduction to NumPy arrays and Matplotlib | Vectors, matrices, indexing, dot products, matrix operations, and linear-model predictions. The practical visualization uses a supplied two-feature dataset with 300 points grouped into three clusters. |
| [Lab 2 - Regression](./Lab%202%20-%20Regression/Lab%202%20%28Part%20A%29%20-%20Introduction%20to%20gradient%20descent.ipynb) ([report](./Lab%202%20-%20Regression/Lab_2_Group_2.pdf)) | [Part B](./Lab%202%20-%20Regression/Lab%202%20%28Part%20B%29%20-%20Linear%20regression%20with%20one%20feature.ipynb), [Part C](./Lab%202%20-%20Regression/Lab%202%20%28Part%20C%29%20-%20Linear%20regression%20with%20multiple%20features.ipynb), [Part D](./Lab%202%20-%20Regression/Lab%202%20%28Part%20D%29%20-%20Nonlinear%20regression.ipynb) | Gradient descent, cost functions, feature scaling, univariate and multivariate linear regression, normal equations, and nonlinear kernel regression. Exercises use a food-truck population/profit dataset and housing data (house size and bedrooms to price). |
| [Lab 3 - Classification](./Lab%203%20-%20Classification/Lab%203%20-%20Part%20A%20-%20Classification%20with%20logistic%20regression.ipynb) ([report](./Lab%203%20-%20Classification/ML_LAB_3_GROUP_2.pdf)) | [Part B: k-nearest neighbours](./Lab%203%20-%20Classification/Lab%203%20-%20Part%20B%20-%20Classification%20with%20kNN.ipynb) | Binary classification, logistic regression, sigmoid and cost functions, feature mapping, decision boundaries, and kNN. The examples classify university admission from two exam scores and microchip quality using two test measurements. |
| [Lab 4 - Random Forest](./Lab%204%20and%205%20-%20Generalization/Lab%204%20-%20Random%20Forest.ipynb) ([combined report](./Lab%204%20and%205%20-%20Generalization/ML_Lab4_5_Group2.pdf)) |  | Decision trees, random feature/sample selection, random forests, and majority voting. The classifier predicts university admission from two exam scores. |
| [Lab 5 - Generalization and Regularization](./Lab%204%20and%205%20-%20Generalization/Lab%205%20-%20Part%20A%20-%20Regularized%20Linear%20Regression.ipynb) ([combined report](./Lab%204%20and%205%20-%20Generalization/ML_Lab4_5_Group2.pdf)) | [Part B: regularized logistic regression](./Lab%204%20and%205%20-%20Generalization/Lab%205%20-%20Part%20B%20-%20Regularized%20Logistic%20Regression.ipynb) | Regularization, bias and variance, model complexity, and generalization. The regression task predicts dam water flow from reservoir water-level changes; the classification task distinguishes accepted and rejected microchips using nonlinear feature mapping. |
| [Lab 6 - Support Vector Machines](./Lab%206%20-%20SVM/Lab%206%20%28Part%20A%29%20-%20Support%20Vector%20Machines.ipynb) ([report](./Lab%206%20-%20SVM/ML_Lab6_Group2.pdf)) | [Part B: spam classification](./Lab%206%20-%20SVM/Lab%206%20%28Part%20B%29%20-%20Spam%20Classification.ipynb) | Maximum-margin classification, linear and Gaussian-kernel decision boundaries, and tuning `C` and `sigma`. Part A uses synthetic classification datasets; Part B processes email text into vocabulary-based features to classify spam. |
| [Lab 7 - Neural Networks](./Lab%207%20-%20ANN/ML_LAB7_GR2.ipynb) ([report](./Lab%207%20-%20ANN/ML_Lab7_Group2.pdf)) |  | Multilayer neural networks, feedforward propagation, sigmoid activations, one-hot labels, cross-entropy cost, backpropagation, gradient descent, regularization, and hidden-unit visualization. See the detailed overview below. |
| [Lab 8 - K-means Clustering](./Lab%208%20-%20Clustering/ML_Lab8_Group2.ipynb) ([report](./Lab%208%20-%20Clustering/ML_Lab8_Group2.pdf)) |  | Unsupervised learning, centroid assignment and updates, and color quantization. The exercises cluster a two-dimensional dataset and compress a supplied 128 x 128 bird image using palettes of 16, 5, and 2 colors. |

Labs 4 and 5 share a folder and a combined report. Lab 6 also contains copies of its notebooks under `College/`; the links above point to the primary copies.

## Lab 7 (HIGHLIGHTS): Handwritten-digit recognition

Lab 7 applies a feedforward neural network to classify handwritten digits from 0 to 9. Each image is represented as 400 pixel values (a flattened 20 x 20 image), and the supplied dataset contains 5,000 examples. The network maps these 400 inputs through a hidden layer to 10 output units, one for each digit class.

The notebook is organized into two parts:

1. **Feedforward and cost:** Load and inspect the digit data, uses principal component analysis (PCA) to visualize it in two dimensions, implemented the forward pass through the network, and calculated unregularize and regularize costs using provided weights. The cost was found to beabout 0.15835 and a classification accuracy was 98.62% for those pretrained weights.
2. **Backpropagation and training:** Initializes the network weights, compute prediction errors and gradients by propagating errors from the output layer back through the hidden layer, and updated the weights with gradient descent. On comparing the effect of regularization and training iterations, predictions were evaluated, and reshaped the hidden units' 400 input weights into 20 x 20 images to inspect the patterns they learn.

The network have 25 hidden units trained for up to 2,000 iterations. With regularization strength `lambda = 20`, it found approximately 93.86% training accuracy. The unregularized version had higher training accuracy, illustrating why training accuracy alone does not establish generalization. Visualizing hidden units across different regularization strengths shows how weight penalties can change noisy, overfit-looking patterns into more distinct strokes and edges, while stopping too early can leave the units undertrained.

## Conclusion

Together, these labs build a practical foundation in machine learning. They connect the underlying mathematics to working Python implementations, use real and synthetic data to explore model behavior, and reinforce the value of visualizing results and evaluating generalization. The sequence ends by applying neural networks to image classification and K-means to clustering and image compression, demonstrating how different learning approaches address different kinds of problems.