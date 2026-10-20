---
name: write-issue
description: Write or revise the description of a Jira epic or issue. Use whenever asked to create, draft, write, describe, flesh out or rewrite an epic, story, task or bug, including "make an epic for X", "create an issue for this" or "update the description of CLIM-123".
---

# Writing epics and issues

A good description comes from discussion with the human, not from the
agent's imagination. Descriptions go bad when they are written from a
one-sentence prompt and the gaps are filled with guesses. This skill is about
avoiding that. For Jira mechanics (acli, components, `ch-obj-*` labels,
sprints) follow `.claude/commands/jira.md`.

## Rules

1. **Only write what you know.** Every statement must come from what the
   human said or pasted, an issue you have read, or code you have looked at
   in this session. Do not invent goals, numbers, dates, owners, user
   groups, dependencies, designs or acceptance criteria.
2. **Open questions are fine.** If something is undecided, write it down as
   an open question. Do not pick a solution or guess the decision, and do
   not phrase a guess as a plan.
3. **Ask before drafting.** Ask the human to paste whatever context they
   have (office discussions, emails, Slack threads, related issues). Then
   ask short batches of questions until the core questions below are
   answered, or the human says they are open.
4. **Fall back to a placeholder.** If there is too little to go on, create
   the issue with a title and a one-line placeholder body, for example
   "Description to be written by <name>", or "Scope not yet defined" if no
   one is named. A placeholder is better than a made-up description.
5. **Draft first.** Show the full draft (title, description, type,
   component, labels, parent) and wait for approval before creating or
   editing anything in Jira. Never create an epic without an approved draft.
6. **Check your sources.** Before showing a draft, read any issue you refer
   to and any file you name, so keys, paths and claims are correct.

## Core questions

- **Why:** what problem does this solve, and for whom?
- **What:** what is in scope, and what is explicitly out of scope?
- **Done when:** how will we know it is finished?
- **Show to (epics, when relevant):** who will we show the finished work to
  for feedback? Name a person, a user group or a country team. It is often
  the same as "for whom" under Why, but it should be someone we can actually
  reach. Ask about it for epics with a user-facing or stakeholder-driven
  outcome. Skip it for internal work such as refactors, CI or test cleanup.

For an issue, a short answer to each is often enough. For an epic, the first
three need a clear answer or an explicit "open".

## Epic format

Use these sections, leaving out any that would be empty:

```
## Goal            (or ## Why, as bullets of problems)
## Background      context the reader needs; can be merged into Goal
## Scope           (or ## What) bullets of the work, with issue keys
## Out of scope
## Open questions
## Done when       checkable outcomes, not activities
## Related         issue keys with a few words each
## Appendix        optional, see below
```

The sections above the appendix should be precise and to the point, so a
reader gets the goal, scope and done-when in a minute or two. Longer
material that is useful to keep, such as pasted discussions, design notes,
alternatives considered or data, goes in an `## Appendix` at the end and is
referred to from the main text where relevant.

When the epic has a "Show to" answer, include it as a Done when item, for
example "Shown to <who>, and their feedback recorded in this epic or a
follow-up issue". If it is relevant but nobody is known yet, list it under
Open questions instead of guessing a name.

When revising an epic after discussion, add a `## Changes from the previous
version (<date>)` section listing what changed and why.

Exemplars: CLIM-1349 (developer experience, short and goal-driven) and
CLIM-1298 (baseline models, detailed and revised once). Read them for tone
and level of detail; do not copy their content.

## Roadmap initiatives

The team roadmap is the Confluence page [Objectives Oct 2026-March 2027](https://dhis2.atlassian.net/wiki/spaces/Climate/pages/2064220168)
(page id `2064220168`, Climate space). It lists initiatives under each
objective, with the epic keys that implement them or "No epic yet". The page
is the overview, so the link goes from the page to the epic. This link is
from 2026-10-20 and changes when the roadmap is revised.

For every new epic:

1. Read the page and find the initiative(s) the epic belongs to. The epic's
   `ch-obj-*` label points to the objective. Put the proposed initiative in
   the draft for the human to confirm. If none fits, say so; do not force a
   match.
2. After the epic is created, show the changed initiative line (the epic key
   added, replacing "No epic yet" if present) and edit the page only after
   the human approves that edit.

## Issue format

No fixed structure. A few sentences on what and why, and a "Done when" or
acceptance criteria list when it helps. Bugs should say how to reproduce,
what happens and what was expected, but only with details the human gave or
you verified.

## Style

- Short, plain sentences. No filler, marketing language or emojis.
- Be concrete: name files, commands, endpoints and issue keys you have
  checked.
- Do not repeat the title in the first sentence.
- Do not add estimates, priorities, dates or assignees unless the human
  gave them.
