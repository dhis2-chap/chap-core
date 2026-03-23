## Workflow

1. Always run `make lint` and `make test` after adding a feature or fixing a bug
2. Use `uv run` when running tests and CLI commands (e.g., `uv run pytest`, `uv run chap`)
3. No commit attribution (no co-author or generated-by lines)
4. Use conventional commits format for commit messages and PR titles
5. Branch naming should follow conventional commit prefixes: `feat/`, `fix/`, `docs/`, `refactor/`, etc.
6. No emojis in commit messages, PR descriptions, or code comments
7. Be concise when adding code/features. Don't add stuff not specifically related to the problem/task, and avoid nice-to-have extras
8. If you feel that the prompt is bad/unclear, always ask follow-up questions until you have high confidence you will be able to solve the problem given
9. When adding new features or fixes, always add a test. Never access private fields or methods (starting with underscore) of a class through testing. Also, when changing code, if there are no relevant tests, consider adding a simple test for the change.
10. When we ask you to go through some change (or finalize changes), always follow the rules in this document strictly.
11. When making pr that has design document only, don't use docs in title.
12. When writing tests, avoid creating new test data inline. Use existing fixtures from conftest.py files whenever possible. Only create new fixtures in conftest.py if testing edge cases not covered by existing fixtures. This improves test maintainability and reduces duplication.
13. Never access private variables
14. When creating Jira issues, always set at least one component
