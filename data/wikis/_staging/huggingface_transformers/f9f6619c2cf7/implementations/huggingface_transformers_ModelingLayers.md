# Modeling Layers - HuggingFace Transformers

## Metadata

| Property | Value |
|----------|-------|
| Source | `src/transformers/modeling_layers.py` |
| Repository | huggingface/transformers |
| Commit Hash | f9f6619c2cf7 |
| Domain | Machine Learning / Deep Learning |
| Primary Language | Python |
| License | Apache License 2.0 |
| Last Updated | 2025-12-18 |

## Overview

The `modeling_layers.py` module provides reusable base layers and task-specific model heads for the Transformers library. It implements gradient checkpointing functionality and generic task-specific classes for sequence classification, question answering, and token classification that can be mixed into any transformer architecture.

## Description

This module serves as a foundation for building task-specific transformer models by providing:

1. **GradientCheckpointingLayer**: A base class that enables gradient checkpointing to reduce memory usage during training by recomputing intermediate activations during the backward pass instead of storing them.

2. **GenericForSequenceClassification**: A mixin class that adds sequence classification capabilities to any transformer model, handling padding tokens and computing pooled logits from the last non-padded token.

3. **GenericForQuestionAnswering**: A mixin class for span-based question answering that predicts start and end positions of answer spans in the input text.

4. **GenericForTokenClassification**: A mixin class for token-level classification tasks (e.g., NER, POS tagging) that applies a classification head to each token's hidden state.

The module uses automatic documentation generation through decorators and supports both tuple and structured output formats. All task-specific classes dynamically instantiate the appropriate base model using `AutoModel.from_config()`, making them architecture-agnostic.

### Key Features

- Gradient checkpointing with automatic cache disabling during training
- Flexible base model prefix support for different architectures
- Proper handling of padding tokens in sequence classification
- Loss computation for all task types when labels are provided
- Support for past key values caching (when not using gradient checkpointing)

## Usage

### Basic Usage

```python
from transformers import AutoConfig
from transformers.modeling_layers import GenericForSequenceClassification

# Create a model configuration
config = AutoConfig.from_pretrained("bert-base-uncased")
config.num_labels = 3

# Instantiate a sequence classification model
# The mixin will automatically create the base model
class MySequenceClassifier(GenericForSequenceClassification, PreTrainedModel):
    pass

model = MySequenceClassifier(config)

# Forward pass
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    labels=labels
)
loss = outputs.loss
logits = outputs.logits
```

### Enabling Gradient Checkpointing

```python
# For any layer that inherits from GradientCheckpointingLayer
class MyTransformerLayer(GradientCheckpointingLayer):
    def forward(self, hidden_states, attention_mask=None):
        # Implementation
        pass

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# During training, checkpointing will be applied automatically
# Note: Pass tensors requiring gradients as positional arguments
output = layer(hidden_states, attention_mask=attention_mask)  # Correct
# output = layer(hidden_states=hidden_states, ...)  # Incorrect with use_reentrant=True
```

### Question Answering

```python
from transformers.modeling_layers import GenericForQuestionAnswering

class MyQuestionAnsweringModel(GenericForQuestionAnswering, PreTrainedModel):
    pass

model = MyQuestionAnsweringModel(config)

outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    start_positions=start_positions,
    end_positions=end_positions
)
loss = outputs.loss
start_logits = outputs.start_logits
end_logits = outputs.end_logits
```

### Token Classification

```python
from transformers.modeling_layers import GenericForTokenClassification

class MyTokenClassifier(GenericForTokenClassification, PreTrainedModel):
    pass

config.num_labels = 9  # Number of token classes
model = MyTokenClassifier(config)

outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    labels=labels
)
loss = outputs.loss
logits = outputs.logits  # Shape: (batch_size, seq_len, num_labels)
```

## Code Reference

### Main Classes

#### GradientCheckpointingLayer

```python
class GradientCheckpointingLayer(nn.Module):
    """Base class for layers with gradient checkpointing.

    Attributes:
        gradient_checkpointing (bool): Whether gradient checkpointing is enabled
    """

    gradient_checkpointing = False

    def __call__(self, *args, **kwargs):
        """
        Handles gradient checkpointing logic and cache management.
        Automatically disables caching when gradient checkpointing is active.
        """
```

#### GenericForSequenceClassification

```python
class GenericForSequenceClassification:
    """Generic sequence classification head for any transformer model.

    Attributes:
        base_model_prefix (str): Name of the base model attribute
        num_labels (int): Number of classification labels
        score (nn.Linear): Classification head
    """

    def __init__(self, config):
        """Initialize with model configuration."""

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> SequenceClassifierOutputWithPast:
        """Forward pass for sequence classification."""
```

#### GenericForQuestionAnswering

```python
class GenericForQuestionAnswering:
    """Generic question answering head for span extraction.

    Attributes:
        base_model_prefix (str): Name of the base model attribute
        qa_outputs (nn.Linear): Linear layer outputting start/end logits
    """

    def __init__(self, config):
        """Initialize with model configuration."""

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> QuestionAnsweringModelOutput:
        """Forward pass for question answering."""
```

#### GenericForTokenClassification

```python
class GenericForTokenClassification:
    """Generic token classification head for sequence labeling.

    Attributes:
        base_model_prefix (str): Name of the base model attribute
        num_labels (int): Number of token classes
        dropout (nn.Dropout): Dropout layer
        score (nn.Linear): Classification head
    """

    def __init__(self, config):
        """Initialize with model configuration and dropout."""

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> TokenClassifierOutput:
        """Forward pass for token classification."""
```

### Imports

```python
from functools import partial
from typing import Optional

import torch
import torch.nn as nn

from .cache_utils import Cache
from .modeling_outputs import (
    BaseModelOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from .models.auto import AutoModel
from .processing_utils import Unpack
from .utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
```

## I/O Contracts

### GradientCheckpointingLayer

#### Inputs
| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| args | tuple | Positional arguments to forward pass | Required |
| kwargs | dict | Keyword arguments to forward pass | Required |

#### Outputs
| Field | Type | Description |
|-------|------|-------------|
| output | Any | Result from the layer's forward pass |

### GenericForSequenceClassification

#### Inputs
| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| input_ids | torch.LongTensor | Input token IDs | None |
| attention_mask | torch.Tensor | Attention mask | None |
| position_ids | torch.LongTensor | Position IDs | None |
| past_key_values | Cache | Cached key/values | None |
| inputs_embeds | torch.FloatTensor | Input embeddings | None |
| labels | torch.LongTensor | Classification labels | None |
| use_cache | bool | Whether to use KV cache | None |

#### Outputs
| Field | Type | Description |
|-------|------|-------------|
| loss | torch.FloatTensor | Classification loss (if labels provided) |
| logits | torch.FloatTensor | Classification logits (batch_size, num_labels) |
| past_key_values | tuple | Cached key/value states |
| hidden_states | tuple | Hidden states from all layers (if output_hidden_states=True) |
| attentions | tuple | Attention weights (if output_attentions=True) |

### GenericForQuestionAnswering

#### Inputs
| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| input_ids | torch.LongTensor | Input token IDs | None |
| attention_mask | torch.Tensor | Attention mask | None |
| position_ids | torch.LongTensor | Position IDs | None |
| past_key_values | Cache | Cached key/values | None |
| inputs_embeds | torch.FloatTensor | Input embeddings | None |
| start_positions | torch.LongTensor | Start positions for training | None |
| end_positions | torch.LongTensor | End positions for training | None |

#### Outputs
| Field | Type | Description |
|-------|------|-------------|
| loss | torch.FloatTensor | QA loss (if positions provided) |
| start_logits | torch.FloatTensor | Start position logits (batch_size, seq_len) |
| end_logits | torch.FloatTensor | End position logits (batch_size, seq_len) |
| hidden_states | tuple | Hidden states from all layers |
| attentions | tuple | Attention weights |

### GenericForTokenClassification

#### Inputs
| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| input_ids | torch.LongTensor | Input token IDs | None |
| attention_mask | torch.Tensor | Attention mask | None |
| position_ids | torch.LongTensor | Position IDs | None |
| past_key_values | Cache | Cached key/values | None |
| inputs_embeds | torch.FloatTensor | Input embeddings | None |
| labels | torch.LongTensor | Token labels | None |
| use_cache | bool | Whether to use KV cache | None |

#### Outputs
| Field | Type | Description |
|-------|------|-------------|
| loss | torch.FloatTensor | Token classification loss (if labels provided) |
| logits | torch.FloatTensor | Token logits (batch_size, seq_len, num_labels) |
| hidden_states | tuple | Hidden states from all layers |
| attentions | tuple | Attention weights |

## Usage Examples

### Example 1: Custom Sequence Classification Model

```python
from transformers import PreTrainedModel, AutoConfig
from transformers.modeling_layers import GenericForSequenceClassification

class CustomBERTForSequenceClassification(GenericForSequenceClassification, PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

# Load configuration and create model
config = AutoConfig.from_pretrained("bert-base-uncased")
config.num_labels = 2
model = CustomBERTForSequenceClassification(config)

# Training
model.train()
outputs = model(
    input_ids=batch["input_ids"],
    attention_mask=batch["attention_mask"],
    labels=batch["labels"]
)
loss = outputs.loss
loss.backward()

# Inference
model.eval()
with torch.no_grad():
    outputs = model(
        input_ids=test_inputs["input_ids"],
        attention_mask=test_inputs["attention_mask"]
    )
    predictions = torch.argmax(outputs.logits, dim=-1)
```

### Example 2: Memory-Efficient Training with Gradient Checkpointing

```python
class MemoryEfficientLayer(GradientCheckpointingLayer):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.ffn = FeedForward(config)

    def forward(self, hidden_states, attention_mask=None):
        # This forward will be recomputed during backward if checkpointing is enabled
        attn_output = self.attention(hidden_states, attention_mask=attention_mask)
        output = self.ffn(attn_output)
        return output

# Enable gradient checkpointing for the model
model.gradient_checkpointing_enable()

# Training proceeds normally but uses less memory
for batch in dataloader:
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()  # Intermediate activations are recomputed here
    optimizer.step()
```

### Example 3: Multi-Task Model with Different Heads

```python
from transformers.modeling_layers import (
    GenericForSequenceClassification,
    GenericForTokenClassification
)

class MultiTaskModel(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.shared_encoder = AutoModel.from_config(config)

        # Separate heads for different tasks
        self.sequence_classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.token_classifier = nn.Linear(config.hidden_size, config.num_token_labels)

    def forward(self, input_ids, task="sequence", labels=None, **kwargs):
        encoder_outputs = self.shared_encoder(input_ids, **kwargs)
        hidden_states = encoder_outputs.last_hidden_state

        if task == "sequence":
            logits = self.sequence_classifier(hidden_states[:, 0])  # Use [CLS] token
        else:  # token classification
            logits = self.token_classifier(hidden_states)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return {"loss": loss, "logits": logits}
```

### Example 4: Question Answering with Custom Post-Processing

```python
from transformers.modeling_layers import GenericForQuestionAnswering

class MyQAModel(GenericForQuestionAnswering, PreTrainedModel):
    pass

model = MyQAModel.from_pretrained("bert-base-uncased")

# Inference
context = "The Transformers library was created by HuggingFace in 2018."
question = "When was Transformers created?"

inputs = tokenizer(question, context, return_tensors="pt")
outputs = model(**inputs)

# Get the most likely answer span
start_idx = torch.argmax(outputs.start_logits)
end_idx = torch.argmax(outputs.end_logits)

if start_idx <= end_idx:
    answer_tokens = inputs.input_ids[0][start_idx:end_idx + 1]
    answer = tokenizer.decode(answer_tokens)
    print(f"Answer: {answer}")  # Output: "2018"
```

## Related Pages

- (To be populated)
