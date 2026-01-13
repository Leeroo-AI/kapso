# Principle: Data_Preparation

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Blog|Karpathy Micrograd Demo|https://github.com/karpathy/micrograd/blob/master/demo.ipynb]]
* [[source::Textbook|Deep Learning Book Ch. 5|https://www.deeplearningbook.org/contents/ml.html]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Data_Science]]
|-
! Last Updated
| [[last_updated::2026-01-13 12:00 GMT]]
|}

== Overview ==

Fundamental principle of organizing raw data into structured input-output pairs suitable for neural network training.

=== Description ===

Data Preparation is the foundational step in any machine learning pipeline. It involves:

1. **Format Conversion:** Transforming raw data into numerical representations that neural networks can process
2. **Input-Output Pairing:** Creating corresponding pairs of feature vectors (inputs) and target values (labels/outputs)
3. **Numerical Encoding:** Ensuring all values are numeric (floats or integers) since neural networks operate on continuous functions

In micrograd, data is represented as simple Python lists:
- **Inputs (`xs`):** A list of input vectors, where each vector is a list of floats
- **Outputs (`ys`):** A list of target values (scalars), one per input vector

This principle is library-agnostic—the same concept applies to PyTorch DataLoaders, TensorFlow Datasets, and any other ML framework.

=== Usage ===

Apply this principle when:
- Starting a new neural network training project
- Converting domain-specific data (images, text, tabular) into numerical format
- Preparing data for the forward pass of a model

The data format directly impacts the network architecture: the input vector length determines the number of input neurons, and the output format determines the output layer configuration.

== Theoretical Basis ==

The mathematical foundation is the concept of a training set:

<math>
\mathcal{D} = \{(x^{(1)}, y^{(1)}), (x^{(2)}, y^{(2)}), ..., (x^{(n)}, y^{(n)})\}
</math>

Where:
- <math>x^{(i)} \in \mathbb{R}^d</math> is the i-th input feature vector of dimension d
- <math>y^{(i)}</math> is the corresponding target value (scalar for regression, class label for classification)
- <math>n</math> is the number of training examples

'''Pseudo-code Pattern:'''
<syntaxhighlight lang="python">
# Abstract data preparation pattern
inputs = []   # List of feature vectors
outputs = []  # List of target values

for raw_sample in raw_data:
    features = extract_features(raw_sample)  # Convert to numeric vector
    label = extract_label(raw_sample)        # Get target value
    inputs.append(features)
    outputs.append(label)
</syntaxhighlight>

== Practical Guide ==

For micrograd, data preparation follows this concrete pattern:

=== Binary Classification Example ===
<syntaxhighlight lang="python">
# Input: 2D points for binary classification
xs = [
    [2.0, 3.0],   # Point 1: features
    [-1.0, -1.0], # Point 2: features
    [3.0, -2.0],  # Point 3: features
    [0.5, 1.0],   # Point 4: features
]

# Output: Binary labels (-1 or 1 for SVM-style, 0 or 1 for sigmoid)
ys = [1.0, -1.0, -1.0, 1.0]  # Corresponding labels
</syntaxhighlight>

=== Key Considerations ===
1. **Input dimension consistency:** All input vectors must have the same length
2. **Data type:** Use floats, not integers, to ensure differentiability
3. **Label encoding:** Match the loss function (e.g., -1/+1 for hinge loss, 0/1 for cross-entropy)
4. **Normalization:** Consider scaling inputs to similar ranges for stable training

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Karpathy_Micrograd_Data_Preparation_Pattern]]
