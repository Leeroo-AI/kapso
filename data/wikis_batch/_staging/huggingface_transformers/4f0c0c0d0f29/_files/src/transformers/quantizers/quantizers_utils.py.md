**Status:** ✅ Explored

**Purpose:** Provides utility functions for quantizer implementations to navigate module hierarchies and determine module conversion eligibility based on exclusion patterns.

**Mechanism:** The get_module_from_name() function splits a dot-separated tensor_name on the last dot to extract module_name and parameter name, then uses module.get_submodule() to retrieve the parent module and returns both the module and tensor name. The should_convert_module() function evaluates whether a module should be quantized by checking if any exclusion patterns match the full_name through three criteria: pattern as prefix with dot, regex match, or suffix match. It returns False (should not convert) if any pattern matches via re.match() for prefix/regex or endswith() for suffix matching.

**Significance:** These utilities abstract common operations needed across all quantizers, enabling consistent module navigation and selective quantization logic. The flexible pattern matching in should_convert_module() supports various exclusion strategies (specific layers, layer patterns, module types), allowing quantizers to skip sensitive components like embeddings, layer norms, or specific attention heads. This centralized logic reduces code duplication and ensures consistent behavior across different quantization methods.
