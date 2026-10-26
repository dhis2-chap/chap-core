## Documentation

When writing documentation with code examples:
1. All `python` code blocks are automatically tested by mktestdocs with `memory=True`, meaning code blocks within a file share state (imports, variables).
2. Structure documentation with imports in earlier blocks, then usage examples in subsequent blocks that can reference those imports.
3. Use real module paths and class names that exist in the codebase.
4. For examples that reference user-created code (like `my_custom_metric.py`), use existing implementations instead (e.g., `MAEMetric` from `chap_core.assessment.metrics.mae`).
5. Only use `console` blocks as a last resort for pseudo-code, CLI commands, or incomplete code signatures that cannot be made executable.
6. When showing class/function signatures, prefer a complete minimal example over an incomplete signature snippet.
7. To render code output in the built docs, use `exec="on" session="<name>" source="above"` on Python code blocks. Add `result="text"` for plain-text output, or omit it when the block prints markdown (e.g. `to_markdown()` tables). Blocks sharing a `session` share state like mktestdocs `memory=True`.
