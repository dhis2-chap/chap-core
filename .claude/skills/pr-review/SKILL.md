---
name: pr-review
description: Review a pull request in this repository. Use whenever asked to review, check, look over, or give feedback on a PR, a branch, or a PR number, including "can you review 607" or "have a look at this PR".
---

# PR review

Follow `docs/contributor/pr_review.md`. Read it once before the first review in
a session. The short version: keep unsafe code, critical bugs, unexamined
agent decisions, and unmaintainable code out of master. Do not polish.

## Steps

1. Identify the PR. If given a number, use it. If given a branch or nothing,
   find the open PR with `gh pr list --head <branch>` or `gh pr view`.
2. Read the PR description with `gh pr view <number>`. Find the
   `Review needed` line and pick the depth:
   - `quick-merge` or missing: run `/code-review <number>` at low effort.
   - `check: <what>`: run `/code-review <number>` at medium effort, then read
     the named area yourself and answer the author's question directly.
   - `full`: run `/code-review <number>` at high effort.
   If the user asks for a specific effort level, that overrides the line.
3. Read the automatic Claude review comment on the PR if there is one. Do not
   repeat its findings. Mention only if you disagree with one.
4. Check the repository conventions CI cannot catch:
   - tests that access private members (names starting with an underscore)
   - new behaviour without a test
   - redundant tests, as defined in the guide
   - a changed `@backtest_plot` registration without the regenerated
     `chap_core/cli_endpoints/generated_plot_ids.py`
   - a new package-root re-export added eagerly instead of in the lazy table in
     `chap_core/__init__.py`
   - a PR title that is not a conventional commit, or emojis anywhere
5. If the user asked you to fix things, or the depth is `quick-merge`, fix
   small non-behavioural issues directly on the branch and say what you
   changed. Never change behaviour or public interfaces without asking.

## Report

End with a short report the reviewer can paste into the PR:

- One line per finding: file and line, the failure scenario in one or two
  sentences, and whether you fixed it.
- One verdict line: safe to merge, needs a look, or do not merge.
- Nothing else. No praise, no summary of what the PR does.

If asked to post the review, use `gh pr review <number> --comment --body` or
`--request-changes` for a do-not-merge verdict. Do not approve on behalf of a
human.
