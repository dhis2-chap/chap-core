# Jira Reference

Working with Jira for the CHAP project. Read the **Tooling** and **Site
knowledge** sections first; they apply regardless of which tool you use.

## Tooling

Pick the tool based on what is available in the current session:

1. **Atlassian MCP** – if tools named `mcp__atlassian__*` are in your tool list
   (for example `searchJiraIssuesUsingJql`, `getJiraIssue`, `createJiraIssue`,
   `editJiraIssue`, `transitionJiraIssue`, `addOrEditJiraIssueComment`), use
   them. Call `getAccessibleAtlassianResources` once to get the `cloudId` and
   reuse it. Custom fields (sprint, story points) are omitted from the default
   compact view; request `view: full` or the specific `customfield_*` ids.
2. **acli** – otherwise use the Atlassian CLI via Bash. See the acli reference
   further down. Check auth with `acli auth status`, log in with
   `acli auth login`.

The MCP is configured per user, not in the repo, so never assume it exists.

Where the two differ: with the MCP, `createJiraIssue` and `editJiraIssue`
accept components, parent, sprint and assignee directly as fields, so the acli
workarounds below (`--from-json`, no custom fields on edit, assign by email)
do not apply. The site knowledge (project, board, sprint field id, component
names, people) is the same for both.

## Site knowledge

- **Project**: always `CLIM` unless told otherwise. Default all queries and
  new issues to it, and read "123" as `CLIM-123`.
- **Component is required**: every new issue must have at least one
  component. Names are case-sensitive; the main one is
  `Chap Modeling Platform`. List components in use:
  `project = CLIM AND component is not EMPTY`.
- **Objective label is required**: every new issue must carry at least one
  `ch-obj-*` label from the list below. See "Objective labels".
- **Board**: `686` is the C&H scrum board, the only scrum board for CLIM.
  Epics can be in sprints on it.
- **Sprint field**: `customfield_10020`. Find the active sprint id from the
  board (see acli commands below, or the MCP board/sprint operations via
  `discover`).
- **Issue types**: Task, Story, Bug, Epic.
- **Statuses**: "To Do", "In Progress", "Done".

### Objective labels

Every new CLIM issue must have at least one `ch-obj-*` label. The labels map
to the objectives in the team roadmap (autumn 2026). The list below is from
2026-10-06 and is revised when the roadmap is; if an issue fits none of them,
ask rather than inventing a new label.

Pick every label that applies. More than one is common, for example a
benchmarking feature is `ch-obj-scientific-evidence` and
`ch-obj-platform-maturity`. Use the exact spelling; labels are free text in
Jira and typos create new labels.

| Label | Use for |
|-------|---------|
| `ch-obj-early-warning` | Getting CHAP to a functional end-to-end early warning system: alerts, scheduling, notifications, thresholds, monitoring, the modeling app and dashboards, demos, and orchestration or integration with DHIS2 and Caps. |
| `ch-obj-scientific-evidence` | Showing that what we build works: papers, benchmarks and the leaderboard, continuous evaluation against reality, evaluation correctness, and documenting rationale and evidence for functionality. |
| `ch-obj-platform-maturity` | Making CHAP something outsiders can install, trust and contribute to: setup and DevOps, the model marketplace, security, documentation, safer formats such as safetensors, and reducing technical debt. |
| `ch-obj-explainability` | Things built on top of a prediction: why this alert or prediction, surfacing model quality in the modeling app, transparency about what the model does, counterfactual explanations. |
| `ch-obj-new-domains` | Growing beyond spatiotemporal outbreak forecasting: tabular data support, intervention effects, and other new data types or analysis questions. |
| `ch-obj-model-quality` | Improving the models themselves: getting a model that is good enough and defining what that means, seasonal forecasts as input, model performance work. |

Work process and team health have no label of their own; tag such work with
the objective it serves, usually `ch-obj-platform-maturity`.

Find issues missing an objective label:
`project = CLIM AND created >= 2026-10-06 AND (labels is EMPTY OR labels not in (ch-obj-early-warning, ch-obj-scientific-evidence, ch-obj-platform-maturity, ch-obj-explainability, ch-obj-new-domains, ch-obj-model-quality))`

With the MCP, pass `labels` to `createJiraIssue` or `editJiraIssue`. With
acli, put `"labels": ["ch-obj-early-warning"]` in the `--from-json` file, or
check `acli jira workitem create --help` for a label flag.

### Team members

| Name | Email |
|------|-------|
| Ivar Grytten | `ivar@dhis2.org` |
| Knut Dagestad Rand | `knut.rand@dhis2.org` |
| Morten Hansen | `morten@dhis2.org` |
| Edvin Aamot Stava | `edvin@dhis2.org` |
| Boris Simovski | `boris@dhis2.org` |
| Eirik Haugstulen | `eirik@dhis2.org` |
| Abyot Asalefew Gizaw | `abyot@dhis2.org` |

The pattern is `firstname@dhis2.org`; Knut is the exception. All verified.
Two account-ID formats coexist: `712020:`-prefixed UUIDs (Ivar, Knut, Edvin,
Boris) and bare hex (Morten, Eirik, Abyot). Both are valid. Always verify the
assignee after assigning.

## Working conventions

- **Creating issues**: infer or ask for type, summary, description and
  priority. Use actionable summaries, add acceptance criteria when useful,
  set a component, set one or more objective labels, and link related
  issues or a parent epic.
- **Querying**: present key, summary, status and assignee in a table, group
  logically (status, epic, priority), and highlight blockers.
- **Relating code to tickets**: use `git log` and branch names
  (`feat/`, `fix/`, `docs/`, `refactor/`) and match `CLIM-\d+` patterns.
  Report completed, in-progress and unlinked changes separately.
- Use JQL for anything beyond a single-issue lookup.

## Useful JQL patterns

- **Assigned to me**: `assignee = currentUser()`
- **Recent updates**: `ORDER BY updated DESC`
- **Specific project**: `project = CLIM`
- **Status filter**: `status = "In Progress"`
- **Multiple conditions**: `project = CLIM AND assignee = currentUser() AND status != Done`
- **Current sprint**: `project = CLIM AND sprint in openSprints()`

## acli reference

Everything below applies only when using the CLI.

### Search and view

```bash
acli jira workitem search --jql "YOUR_JQL_QUERY" --limit 10
acli jira workitem search --jql "assignee = currentUser() ORDER BY updated DESC" --limit 10
acli jira workitem view CLIM-123
```

### Create

```bash
acli jira workitem create --project CLIM --type Task --summary "Summary text" --description "Detailed description"
```

#### Create with a component, parent or sprint

`acli` has no `--component` flag. To set a component, parent or sprint, create
the work item from a JSON file with `--from-json`. These must live under
`additionalAttributes`, not at the top level. The description must be in
Atlassian Document Format (ADF).

```json
{
  "projectKey": "CLIM",
  "type": "Task",
  "summary": "Your summary",
  "description": {"type":"doc","version":1,"content":[{"type":"paragraph","content":[{"type":"text","text":"Body"}]}]},
  "additionalAttributes": {
    "components": [{"name": "Chap Modeling Platform"}],
    "parent": {"key": "CLIM-718"},
    "customfield_10020": 2354
  }
}
```

```bash
acli jira workitem create --from-json /tmp/issue.json
```

The sprint can only be set **at create time**; `acli jira workitem edit`
strict-rejects every custom field (`json: unknown field`), so an existing work
item cannot be moved into a sprint with acli. Do that in the board UI or via
the MCP.

### Transition

```bash
acli jira workitem transition --key CLIM-123 --status "Done" --yes
```

### Comments

```bash
acli jira workitem comment create --key "CLIM-123" --body "Your comment text"
acli jira workitem comment list --key "CLIM-123" --limit 5
```

### Boards and sprints

```bash
acli jira board search --project CLIM              # 686 = C&H scrum
acli jira board list-sprints --id 686 --state active,future --csv
acli jira sprint list-workitems --board 686 --sprint 2354 --jql 'component = "Chap Modeling Platform"' --csv
```

`acli jira sprint update` is a full replace: `--name` and `--state` are required
even when you only mean to change the dates.

### Assigning

**Assign by email, never by account ID.** `--assignee` claims to accept an
account ID, but when given one `acli` reports `SUCCESS` and **silently
unassigns the work item instead**, including wiping an existing assignee.
Both ID formats fail this way.

```bash
acli jira workitem assign --key "CLIM-1,CLIM-2" --assignee edvin@dhis2.org --yes
acli jira workitem search --jql 'key in (CLIM-1,CLIM-2)' --json \
  | python3 -c "import sys,json;[print(x['key'],(x['fields'].get('assignee') or {}).get('displayName')) for x in json.load(sys.stdin)]"
```

Account IDs are still needed for the `reporter` field in `--from-json` (where
they do work, as `{"id": "..."}`) and for JQL. Look one up from an issue the
person is assigned to:

```bash
acli jira workitem search --jql 'project=CLIM AND assignee is not EMPTY' --limit 200 --paginate --json \
  | python3 -c "import sys,json;[print(a['displayName'],a['accountId']) for a in {x['fields']['assignee']['accountId']:x['fields']['assignee'] for x in json.load(sys.stdin) if x['fields'].get('assignee')}.values()]"
```

The CSV output renders assignee as an email, which is hidden for everyone but
yourself, so it shows blank. Use `--json` and read `assignee.displayName`.

### Help

- `acli jira --help`, `acli jira workitem --help`, `acli jira workitem search --help`
- ACLI documentation: https://developer.atlassian.com/cloud/acli/
- JQL reference: https://support.atlassian.com/jira-software-cloud/docs/use-advanced-search-with-jira-query-language-jql/
