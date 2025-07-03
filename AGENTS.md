# Agent Instructions

This repository includes several Gradio applications and supporting code.

## Scope
These instructions apply to the entire repository.

## Coding Guidelines
- Use Python 3 syntax with type hints where practical.
- Add docstrings to new functions and classes.
- Keep line length under 120 characters.
- Avoid dynamic code execution (`exec`/`eval`). Prefer explicit loops or helper functions.

## Commit Guidelines
- Commit messages should be short and present tense ("Add feature" not "Added feature").
- Include a brief summary of changes in the PR description under a `## Summary` section.
- Include a `## Testing` section in the PR body describing commands run to verify functionality.

## Validation
- After modifying Python files, run `python -m py_compile <file>` for each changed file to ensure there are no syntax errors.
- If other checks or tests are present in subdirectories, run them as well.
