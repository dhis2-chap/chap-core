# Release Command

Create a new release for chap-core. Follow these steps:

## 1. Find the last release tag and changes

```bash
git tag -l 'v*' --sort=-v:refname | head -1
```

Then get commits since that tag:
```bash
git log <last-tag>..HEAD --oneline
```

## 2. Ask user for the new version

Present the list of changes and ask the user what version number to use (suggest patch/minor/major based on changes).

## 3. Generate release notes

Organize the commits into categories:
- **Highlights**: Major new features (1-2 sentences each)
- **Features**: New functionality
- **Bug Fixes**: Fixes
- **Documentation**: Doc changes (if significant)

Keep the notes concise.

## 4. Update version in codebase

Edit `pyproject.toml` to update the version field (line 3).

Then run:
```bash
uv lock
```

## 5. Commit and push

```bash
git add pyproject.toml uv.lock
git commit -m "chore: bump version to <new-version>"
git push origin master
```

## 6. Create tag and GitHub release

```bash
gh release create v<new-version> --title "v<new-version>" --notes "<release-notes>"
```

Use a HEREDOC for the notes to preserve formatting.

## 7. Trigger PyPI upload

```bash
gh workflow run "Upload to PyPI"
```

Then show the user the workflow status:
```bash
gh run list --workflow="Upload to PyPI" --limit=1
```

Provide the GitHub Actions URL so they can monitor progress.