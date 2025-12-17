# File: `utils/check_modeling_structure.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 150 |
| Functions | `iter_modeling_files`, `colored_error_message`, `full_name`, `check_init_weights`, `is_self_method_call`, `is_super_method_call`, `check_post_init`, `main` |
| Imports | ast, pathlib, rich, sys |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Enforces critical conventions in modeling files: (1) _init_weights must use init.* functions for in-place initialization, and (2) PreTrainedModel subclasses must call self.post_init() in __init__.

**Mechanism:** Parses all modeling_*.py and modular_*.py files using AST, walks the syntax tree to find: (1) _init_weights methods and checks for forbidden in-place operations (*.xxx_()) on module weights, (2) PreTrainedModel subclasses and verifies __init__ contains self.post_init() call or super().__init__() for modular files. Reports violations with colored line-number references.

**Significance:** Architecture enforcement tool that prevents subtle initialization bugs by ensuring weight initialization follows the library's internal flagging system (needed for model merging/adapters) and that post-initialization hooks (like tying weights) execute correctly across all models.
