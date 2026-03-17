## Workflow

1. Always run `make lint` and `make test` after adding a feature or fixing a bug
2. No commit attribution (no co-author or generated-by lines)
3. Use conventional commits format for commit messages and PR titles
4. Branch naming should follow conventional commit prefixes: `feat/`, `fix/`, `docs/`, `refactor/`, etc.
5. No emojis in commit messages, PR descriptions, or code comments
6. Be concise when adding code/features. Don't add stuff not specifically related to the problem/task, and avoid nice-to-have extras
7. If you feel that the prompt is bad/unclear, always ask follow-up questions until you have high confidence you will be able to solve the problem given
8. When adding new features or fixes, always add a test. Never access private fields or methods (starting with underscore) of a class through testing. Also, when changing code, if there are no relevant tests, consider adding a simple test for the change.
9. When we ask you to go through some change (or finalize changes), always follow the rules in this document strictly.
10. When making pr that has design document only, don't use docs in title.
11. When writing tests, avoid creating new test data inline. Use existing fixtures from conftest.py files whenever possible. Only create new fixtures in conftest.py if testing edge cases not covered by existing fixtures. This improves test maintainability and reduces duplication.
12. Never access private variables
13. When creating Jira issues, always set at least one component
