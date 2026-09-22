# Pull request review

Most pull requests in this repository are written with AI coding agents, and
the number of PRs is higher than a team can review line by line. Review is
therefore split between an automatic first pass and a short human pass. This
page describes what each pass is for and how authors and reviewers work
together.

## What review is for

Review exists to keep four things out of master, in this order:

1. Malicious or unsafe code: injection, unsafe deserialisation, leaked
   secrets, unsafe file or subprocess handling.
2. Critical bugs that would reach production, especially edge cases the tests
   do not cover.
3. Surprising agent decisions that nobody has looked at: silently swallowed
   errors, disabled or weakened tests, scope well beyond the PR title, or a
   roundabout solution where a simple one exists.
4. Code that will be hard to maintain.

Review is not for polish. Formatting is handled by ruff in CI, and style
preferences are not worth a review round trip.

## The automatic first pass

Every non-draft PR from this repository is reviewed by Claude through the
`claude-pr-review` GitHub Actions workflow. It runs once when the PR is opened
or marked ready for review, and posts inline comments plus one summary comment
with a verdict: safe to merge, needs a look, or do not merge.

It does not re-run on every push. To re-run it after changes, comment
`@claude review` on the PR.

The human reviewer should read the summary first and treat it as done work.
Do not repeat the checks it already made.

## Asking for review

The PR template has a `Review needed` line. The author fills in one of three
values, and the reviewer adjusts effort accordingly:

- `quick-merge`: the author is confident. The reviewer reads the automatic
  summary, skims the diff, fixes small things directly, and merges.
- `check: <what>`: the author is unsure about one specific thing. The
  reviewer looks at that thing properly and skims the rest.
- `full`: the author wants a thorough review, either because the change is
  risky or because they want to learn from it. The reviewer reads everything
  and explains findings.

If a PR sits for more than a couple of days, the author should ask in Slack and
say which of the three they need.

## Fixing things directly

Reviewers may push small fixes to the author's branch instead of requesting
changes. This covers formatting, naming, typos, docstrings, test cleanups, and
similar changes that do not alter behaviour. Leave a one-line comment on the
PR saying what was changed.

Anything that changes behaviour, public interfaces, or the approach goes back
to the author as a comment, so they can decide and learn from it.

## Redundant tests

Agents tend to add more tests than a change needs. A test is redundant when it
duplicates an existing test, exercises the same code path several times with
trivially different inputs, or tests framework behaviour rather than this
repository's code. Reviewers should point at the overlapping existing test and
remove or ask to remove the redundant one. Fewer, sharper tests are easier to
keep green.

## Reviewing with an agent

The repository ships a `pr-review` skill for Claude Code. Asking Claude to
review a PR in this repository picks it up automatically, or invoke it with
`/pr-review <number>`. It reads the `Review needed` line, runs a correctness
review at matching depth, checks the repository conventions that CI cannot
check, and reports in the same verdict format as the automatic review. A
reviewer who wants an occasional deep pass asks for high effort explicitly.
