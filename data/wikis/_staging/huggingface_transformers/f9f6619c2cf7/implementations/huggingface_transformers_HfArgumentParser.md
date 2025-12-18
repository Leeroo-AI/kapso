{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|huggingface_transformers|https://github.com/huggingface/transformers]]
|-
! Domains
| [[domain::CLI]], [[domain::Utilities]], [[domain::Configuration]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
An enhanced ArgumentParser that automatically generates command-line arguments from Python dataclass type hints, supporting JSON/YAML config files and advanced features like aliases and boolean flags.

=== Description ===
`HfArgumentParser` extends Python's standard `argparse.ArgumentParser` to automatically create CLI arguments based on dataclass field definitions. It uses type hints to determine argument types, default values from field defaults, and help text from field metadata. This eliminates boilerplate code for argument parsing and ensures type safety between dataclass definitions and CLI interfaces.

The parser supports multiple dataclasses simultaneously (useful for separating model args, training args, etc.), can load arguments from JSON/YAML files, handles complex types like `Optional`, `Union`, `Literal`, and `list`, and provides special syntax for boolean arguments. It integrates seamlessly with HuggingFace's training ecosystem.

=== Usage ===
Use this parser when building CLI tools or training scripts that need structured, type-safe argument parsing. It's particularly useful for ML experiments where you want to define all hyperparameters in dataclasses, support both CLI and config file inputs, and maintain consistency between argument definitions and code usage.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers huggingface_transformers]
* '''File:''' src/transformers/hf_argparser.py

=== Signature ===
<syntaxhighlight lang="python">
class HfArgumentParser(ArgumentParser):
    def __init__(
        self,
        dataclass_types: Optional[Union[DataClassType, Iterable[DataClassType]]] = None,
        **kwargs
    ):
        """
        Args:
            dataclass_types: Single dataclass or list of dataclasses to parse
            **kwargs: Passed to ArgumentParser (e.g., description, prog)
        """

    def parse_args_into_dataclasses(
        self,
        args=None,
        return_remaining_strings=False,
        look_for_args_file=True,
        args_filename=None,
        args_file_flag=None
    ) -> tuple[DataClass, ...]:
        """Parse CLI args into dataclass instances."""

    def parse_dict(
        self,
        args: dict[str, Any],
        allow_extra_keys: bool = False
    ) -> tuple[DataClass, ...]:
        """Parse from dictionary instead of CLI args."""

    def parse_json_file(
        self,
        json_file: Union[str, os.PathLike],
        allow_extra_keys: bool = False
    ) -> tuple[DataClass, ...]:
        """Load args from JSON file."""

    def parse_yaml_file(
        self,
        yaml_file: Union[str, os.PathLike],
        allow_extra_keys: bool = False
    ) -> tuple[DataClass, ...]:
        """Load args from YAML file."""

def HfArg(
    *,
    aliases: Optional[Union[str, list[str]]] = None,
    help: Optional[str] = None,
    default: Any = dataclasses.MISSING,
    default_factory: Callable[[], Any] = dataclasses.MISSING,
    metadata: Optional[dict] = None,
    **kwargs
) -> dataclasses.Field:
    """Helper for creating dataclass fields with parser metadata."""

def string_to_bool(v) -> bool:
    """Convert string to boolean (handles yes/no, true/false, etc.)."""

def make_choice_type_function(choices: list) -> Callable[[str], Any]:
    """Create type converter for choice arguments."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from transformers import HfArgumentParser
# or
from transformers.hf_argparser import HfArgumentParser, HfArg
</syntaxhighlight>

== I/O Contract ==

=== Initialization Parameters ===
{| class="wikitable"
! Parameter !! Type !! Default !! Description
|-
| dataclass_types || DataClassType or Iterable || None || Dataclass(es) to generate arguments from
|-
| **kwargs || dict || {} || Arguments passed to ArgumentParser (description, prog, etc.)
|}

=== parse_args_into_dataclasses Parameters ===
{| class="wikitable"
! Parameter !! Type !! Default !! Description
|-
| args || list[str] || None || Argument strings to parse (default: sys.argv)
|-
| return_remaining_strings || bool || False || Return unparsed arguments
|-
| look_for_args_file || bool || True || Look for .args file with script's base name
|-
| args_filename || str || None || Specific args file to load
|-
| args_file_flag || str || None || CLI flag to specify args file (e.g., "--config")
|}

=== Return Values ===
{| class="wikitable"
! Method !! Returns !! Description
|-
| parse_args_into_dataclasses || tuple[DataClass, ...] || Dataclass instances populated with parsed values
|-
| parse_dict || tuple[DataClass, ...] || Dataclass instances from dictionary
|-
| parse_json_file || tuple[DataClass, ...] || Dataclass instances from JSON file
|-
| parse_yaml_file || tuple[DataClass, ...] || Dataclass instances from YAML file
|}

=== HfArg Parameters ===
{| class="wikitable"
! Parameter !! Type !! Description
|-
| aliases || str or list[str] || Alternative CLI flags (e.g., ["-e", "--example"])
|-
| help || str || Help text shown with --help
|-
| default || Any || Default value (mutually exclusive with default_factory)
|-
| default_factory || Callable || Function returning default value (for mutable types)
|-
| metadata || dict || Additional metadata for dataclasses.field
|}

== Usage Examples ==

=== Basic Usage with Dataclass ===
<syntaxhighlight lang="python">
from dataclasses import dataclass
from transformers import HfArgumentParser

@dataclass
class ModelArguments:
    model_name: str
    num_layers: int = 12
    dropout: float = 0.1
    use_cache: bool = True

# Create parser
parser = HfArgumentParser(ModelArguments)

# Parse from command line
# python script.py --model_name bert-base --num_layers 6
model_args, = parser.parse_args_into_dataclasses()

print(model_args.model_name)  # "bert-base"
print(model_args.num_layers)  # 6
</syntaxhighlight>

=== Multiple Dataclasses ===
<syntaxhighlight lang="python">
from dataclasses import dataclass
from transformers import HfArgumentParser

@dataclass
class ModelArgs:
    model_name: str
    hidden_size: int = 768

@dataclass
class TrainingArgs:
    learning_rate: float = 5e-5
    batch_size: int = 32
    num_epochs: int = 3

# Parse both together
parser = HfArgumentParser([ModelArgs, TrainingArgs])
model_args, training_args = parser.parse_args_into_dataclasses()

print(model_args.hidden_size)      # 768
print(training_args.learning_rate) # 5e-5
</syntaxhighlight>

=== Using HfArg for Enhanced Features ===
<syntaxhighlight lang="python">
from dataclasses import dataclass
from transformers import HfArgumentParser
from transformers.hf_argparser import HfArg

@dataclass
class Args:
    # Aliases allow multiple flag names
    model: str = HfArg(
        default="bert-base",
        aliases=["-m", "--model-name"],
        help="Model identifier or path"
    )

    # List arguments
    layers: list[int] = HfArg(
        default_factory=list,
        help="Layer indices to use"
    )

    # Boolean with custom help
    verbose: bool = HfArg(
        default=False,
        help="Enable verbose logging"
    )

parser = HfArgumentParser(Args)

# Can use any alias:
# python script.py -m roberta-base --layers 0 1 2 --verbose
args, = parser.parse_args_into_dataclasses()
</syntaxhighlight>

=== Loading from JSON Config ===
<syntaxhighlight lang="python">
from dataclasses import dataclass
from transformers import HfArgumentParser

@dataclass
class TrainingConfig:
    learning_rate: float
    batch_size: int
    num_epochs: int

# config.json:
# {
#   "learning_rate": 3e-5,
#   "batch_size": 16,
#   "num_epochs": 10
# }

parser = HfArgumentParser(TrainingConfig)
config, = parser.parse_json_file("config.json")

print(config.learning_rate)  # 3e-5
</syntaxhighlight>

=== Loading from YAML Config ===
<syntaxhighlight lang="python">
# config.yaml:
# model_name: gpt2-large
# max_length: 1024
# temperature: 0.7

@dataclass
class GenerationConfig:
    model_name: str
    max_length: int
    temperature: float

parser = HfArgumentParser(GenerationConfig)
config, = parser.parse_yaml_file("config.yaml")
</syntaxhighlight>

=== Combining CLI and Config File ===
<syntaxhighlight lang="python">
from dataclasses import dataclass
from transformers import HfArgumentParser

@dataclass
class Args:
    model: str = "bert-base"
    lr: float = 1e-4
    epochs: int = 3

parser = HfArgumentParser(Args)

# Look for script_name.args file automatically
# CLI args override file args
# python train.py --lr 5e-5
# (train.args contains: --model roberta-base --epochs 10)

args, = parser.parse_args_into_dataclasses(look_for_args_file=True)
# Result: model="roberta-base", lr=5e-5, epochs=10
</syntaxhighlight>

=== Using parse_dict for Programmatic Config ===
<syntaxhighlight lang="python">
@dataclass
class Config:
    param1: str
    param2: int

parser = HfArgumentParser(Config)

# Parse from dictionary
config_dict = {"param1": "value", "param2": 42}
config, = parser.parse_dict(config_dict)

print(config.param1)  # "value"
</syntaxhighlight>

=== Complex Types: Optional, Literal, Enum ===
<syntaxhighlight lang="python">
from dataclasses import dataclass
from typing import Optional, Literal
from enum import Enum

class Precision(Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"

@dataclass
class AdvancedArgs:
    # Optional type
    checkpoint: Optional[str] = None

    # Literal type (choices)
    device: Literal["cpu", "cuda", "mps"] = "cuda"

    # Enum type
    precision: Precision = Precision.FP32

parser = HfArgumentParser(AdvancedArgs)

# python script.py --checkpoint model.pt --device cuda --precision fp16
args, = parser.parse_args_into_dataclasses()
</syntaxhighlight>

=== Boolean Arguments with No-Prefix ===
<syntaxhighlight lang="python">
@dataclass
class BoolArgs:
    # When default is True, automatically creates --no_flag option
    use_cache: bool = True
    verbose: bool = False

parser = HfArgumentParser(BoolArgs)

# Can use:
# --use_cache (sets to True)
# --no_use_cache (sets to False)
# --verbose (sets to True)
# python script.py --no_use_cache --verbose
args, = parser.parse_args_into_dataclasses()
# Result: use_cache=False, verbose=True
</syntaxhighlight>

== Implementation Details ==

=== Type Handling ===
* Uses `typing.get_type_hints()` to resolve string annotations
* Supports `Union[X, None]` (i.e., `Optional[X]`)
* `Union` with multiple non-None types raises error (except with str)
* `Literal` types become choice arguments
* `Enum` types extract `.value` as choices
* `list[T]` types use `nargs="+"` in argparse

=== Argument Name Conventions ===
* Dataclass field `field_name` creates `--field_name` and `--field-name`
* Underscores and hyphens are interchangeable
* Boolean fields with `default=True` auto-generate `--no_field_name` option

=== Config File Precedence ===
When multiple sources provide arguments:
1. Default .args file (script_name.args)
2. Explicitly specified args file
3. Command-line arguments (highest priority)

Later sources override earlier ones.

=== Formatter Class ===
Uses `ArgumentDefaultsHelpFormatter` by default to show defaults in help text.

== Related Pages ==
