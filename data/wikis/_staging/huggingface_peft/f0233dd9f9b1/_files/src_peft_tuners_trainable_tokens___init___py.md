# src/peft/tuners/trainable_tokens/__init__.py

## Overview
This module initializes the Trainable Tokens method for PEFT. Trainable Tokens is a parameter-efficient fine-tuning technique that allows training specific token embeddings while keeping the rest of the embedding matrix frozen. This is particularly useful for adding new tokens or updating existing token embeddings without the memory and computational cost of training the full embedding matrix.

## Key Components

### Imports
- `TrainableTokensConfig`: Configuration class for Trainable Tokens settings
- `TrainableTokensLayer`: Layer wrapper that implements selective token training
- `TrainableTokensModel`: Main model class implementing Trainable Tokens functionality

### Registration
The module registers Trainable Tokens as a PEFT method using `register_peft_method()` with:
- **name**: "trainable_tokens"
- **config_cls**: TrainableTokensConfig
- **model_cls**: TrainableTokensModel
- **is_mixed_compatible**: False (cannot be mixed with other PEFT methods)

## Purpose
Trainable Tokens enables parameter-efficient fine-tuning by:
- Making only specified token embeddings trainable
- Keeping the rest of the embedding matrix frozen
- Reducing both storage and working memory compared to full embedding training
- Supporting addition of new tokens without full model fine-tuning

## Use Cases

### Adding New Tokens
- Add domain-specific vocabulary
- Add special tokens for new tasks
- Extend tokenizer without retraining full embedding

### Updating Existing Tokens
- Fine-tune embeddings for domain adaptation
- Adjust token representations for specific tasks
- Improve embeddings for underperforming tokens

## Exports
The module exports three main classes:
- `TrainableTokensConfig`: Configuration for token selection
- `TrainableTokensLayer`: Layer wrapper for selective token training
- `TrainableTokensModel`: Model class for applying Trainable Tokens

## Limitations

### Not Mixed Compatible
Unlike some PEFT methods (e.g., LoKr), Trainable Tokens cannot be combined with other PEFT methods on the same model. This is because it operates at the token embedding level and may conflict with other adaptation approaches.

### FSDP/DeepSpeed Support
Note in the config indicates that training with FSDP/DeepSpeed might not yet be fully supported. This is an area of ongoing development.

## Integration
This method is typically used when:
- Adding new tokens to vocabulary
- Fine-tuning small number of token embeddings
- Memory constraints prevent full embedding matrix training
- Need to update specific tokens without affecting others
