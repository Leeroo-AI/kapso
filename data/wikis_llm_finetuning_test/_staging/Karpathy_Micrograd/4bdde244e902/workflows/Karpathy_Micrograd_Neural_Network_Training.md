# Workflow: Neural_Network_Training

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Micrograd|https://github.com/karpathy/micrograd]]
* [[source::Doc|Micrograd README|https://github.com/karpathy/micrograd/blob/master/README.md]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Autograd]], [[domain::Educational]]
|-
! Last Updated
| [[last_updated::2026-01-13 12:00 GMT]]
|}

== Overview ==
End-to-end process for training a neural network classifier using micrograd's scalar-valued autograd engine and PyTorch-like neural network API.

=== Description ===
This workflow outlines the standard procedure for training neural networks using the micrograd library. Micrograd implements reverse-mode automatic differentiation (backpropagation) over a dynamically built computation graph of scalar values. The process covers data preparation, network architecture definition, forward pass computation, loss calculation, gradient computation via backpropagation, and iterative parameter updates using gradient descent.

The workflow demonstrates the fundamental concepts of deep learning:
* How computation graphs are constructed implicitly through operations
* How gradients flow backward through the chain rule
* How parameters are updated to minimize loss

=== Usage ===
Execute this workflow when you want to:
* Train a binary classifier on 2D data (e.g., moon dataset, circle dataset)
* Learn the fundamentals of how autograd and backpropagation work
* Understand PyTorch-style neural network APIs at a low level
* Build educational examples demonstrating gradient descent

This workflow is ideal for educational purposes where understanding the mechanics of training is more important than performance or scale.

== Execution Steps ==

=== Step 1: Data Preparation ===

Prepare training data as lists of input-output pairs. Each input is a list of scalar values representing features, and each output is a scalar target label (typically +1 or -1 for binary classification).

'''Key considerations:'''
* Inputs must be Python lists of floats (micrograd wraps them as Value objects internally)
* Labels should be numeric scalars compatible with the loss function
* For binary classification, use +1/-1 labels with SVM-style hinge loss, or 0/1 with other losses
* Dataset size is limited by computational efficiency (scalar operations)

=== Step 2: Network Architecture Definition ===

Create a Multi-Layer Perceptron (MLP) by specifying input dimension and layer sizes. The MLP class automatically constructs layers of neurons with ReLU activations (except the output layer which is linear).

'''What happens:'''
* `MLP(nin, [h1, h2, ..., nout])` creates a network with specified hidden layer sizes
* Each Layer contains Neuron objects with randomly initialized weights in [-1, 1]
* Neurons compute weighted sums with bias, followed by optional ReLU activation
* The final layer has no activation (linear output for classification scores)

'''Architecture hierarchy:'''
  MLP → Layer(s) → Neuron(s) → Value (weights, biases)

=== Step 3: Forward Pass Computation ===

Feed input data through the network to produce predictions. Each forward pass dynamically builds a computation graph of Value objects that tracks all operations.

'''What happens:'''
* Input values flow through each layer sequentially
* Each neuron computes: activation = ReLU(sum(w_i * x_i) + b) or linear
* The computation graph is built implicitly through Value operations
* Output is a Value object (or list) representing the network's prediction

=== Step 4: Loss Computation ===

Calculate a scalar loss value that measures prediction quality. Common choices include SVM max-margin (hinge) loss for binary classification.

'''SVM Hinge Loss formula:'''
  loss = sum(max(0, 1 - y_i * score_i)) for all samples

'''What happens:'''
* Compare predictions against target labels
* Aggregate individual sample losses into a single scalar
* Optionally add L2 regularization on weights
* The loss Value contains the entire computation graph via _prev references

=== Step 5: Backward Pass (Gradient Computation) ===

Call backward() on the loss Value to compute gradients of all parameters with respect to the loss. This implements reverse-mode automatic differentiation.

'''What happens:'''
* Topological sort orders all nodes from inputs to loss
* Initialize loss.grad = 1 (derivative of loss w.r.t. itself)
* Traverse nodes in reverse topological order
* Each node's _backward() closure applies the chain rule to propagate gradients
* After completion, every Value in the graph has its .grad attribute populated

=== Step 6: Parameter Update (Gradient Descent) ===

Update all network parameters by taking a step opposite to the gradient direction. This is the core of stochastic gradient descent (SGD).

'''Update rule:'''
  parameter.data -= learning_rate * parameter.grad

'''What happens:'''
* Iterate over model.parameters() to get all trainable Values (weights and biases)
* Subtract a small multiple (learning rate) of each gradient from the parameter
* Common learning rates: 0.01, 0.05, 0.1 (problem-dependent)
* Zero out gradients before next iteration to prevent accumulation

=== Step 7: Training Loop Iteration ===

Repeat steps 3-6 for multiple epochs until the loss converges or accuracy is satisfactory.

'''Training loop structure:'''
  1. Zero gradients: model.zero_grad()
  2. Forward pass: predictions = [model(x) for x in inputs]
  3. Compute loss: loss = loss_function(predictions, targets)
  4. Backward pass: loss.backward()
  5. Update parameters: for p in model.parameters(): p.data -= lr * p.grad
  6. Repeat for N epochs

'''Key considerations:'''
* Monitor loss decrease to verify learning
* Adjust learning rate if training is unstable or too slow
* Typical training: 20-100 epochs for simple datasets
* Optionally track accuracy on validation data

== Execution Diagram ==
{{#mermaid:graph TD
    A[Data Preparation] --> B[Network Architecture Definition]
    B --> C[Forward Pass Computation]
    C --> D[Loss Computation]
    D --> E[Backward Pass]
    E --> F[Parameter Update]
    F --> G{Converged?}
    G -->|No| H[Zero Gradients]
    H --> C
    G -->|Yes| I[Training Complete]
}}

== GitHub URL ==

The executable implementation will be available at:

[[github_url::PENDING_REPO_BUILD]]

<!-- This URL will be populated by the repo builder phase -->
