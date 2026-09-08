# Jira CLI Commands Reference

Use the Atlassian CLI (acli) to interact with Jira.

## Authentication

Check if authenticated:
```bash
acli auth status
```

Login (opens browser for OAuth):
```bash
acli auth login
```

## Projects

List all projects:
```bash
acli jira project list --limit 20
```

## Work Items (Issues)

### Search Issues

Search with JQL query:
```bash
acli jira workitem search --jql "YOUR_JQL_QUERY" --limit 10
```

My assigned issues:
```bash
acli jira workitem search --jql "assignee = currentUser() ORDER BY updated DESC" --limit 10
```

Issues in a specific project:
```bash
acli jira workitem search --jql "project = PROJECT_KEY ORDER BY updated DESC" --limit 10
```

### View Issue Details

```bash
acli jira workitem view ISSUE-KEY
```

Example:
```bash
acli jira workitem view PROJECT-123
```

### Create Work Item

```bash
acli jira workitem create --project PROJECT_KEY --type Task --summary "Summary text" --description "Detailed description"
```

Available types: Task, Story, Bug, Epic, etc.

#### Create with a component (and/or parent)

`acli` has no `--component` flag. To set a component or parent, create the work
item from a JSON file with `--from-json`. Components and parent must live under
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
    "parent": {"key": "CLIM-718"}
  }
}
```

```bash
acli jira workitem create --from-json /tmp/issue.json
```

Gotchas:
- Component name is case-sensitive. List existing components in use with:
  ```bash
  acli jira workitem search --jql 'project=CLIM AND component is not EMPTY' --limit 3
  ```
- `components` and `parent` belong under `additionalAttributes`, not at the top level.

### Transition Work Item

```bash
acli jira workitem transition --key ISSUE-KEY --status "Status Name" --yes
```

Common statuses: "To Do", "In Progress", "Done"

Example:
```bash
acli jira workitem transition --key PROJECT-123 --status "Done" --yes
```

## Comments

Add a comment to an issue:
```bash
acli jira workitem comment create --key "ISSUE-KEY" --body "Your comment text"
```

List comments on an issue:
```bash
acli jira workitem comment list --key "ISSUE-KEY" --limit 5
```

## Useful JQL Patterns

- **Assigned to me**: `assignee = currentUser()`
- **Recent updates**: `ORDER BY updated DESC`
- **Specific project**: `project = PROJECT_KEY`
- **Status filter**: `status = "In Progress"`
- **Multiple conditions**: `project = PROJECT_KEY AND assignee = currentUser() AND status != Done`

## Common Workflows

### Check your tasks
```bash
acli jira workitem search --jql "assignee = currentUser() ORDER BY updated DESC" --limit 10
```

### Update issue with progress
```bash
acli jira workitem comment create --key "PROJECT-123" --body "Progress update: [your update here]"
```

### View project issues
```bash
acli jira workitem search --jql "project = PROJECT_KEY ORDER BY created DESC" --limit 20
```

## Sprints

The Sprint field is `customfield_10020` on this site. It can only be set **at
create time** via `--from-json`; `acli jira workitem edit` strict-rejects every
custom field (`json: unknown field`), so an existing work item cannot be moved
into a sprint with acli — do that in the board UI.

```json
{
  "projectKey": "CLIM",
  "type": "Task",
  "summary": "Your summary",
  "additionalAttributes": {
    "components": [{"name": "Chap Modeling Platform"}],
    "customfield_10020": 2354
  }
}
```

Epics can be in sprints on the C&H scrum board (board 686) — several are.

Find the current sprint and its id:

```bash
acli jira board search --project CLIM              # 686 = C&H scrum, the only scrum board
acli jira board list-sprints --id 686 --state active,future --csv
acli jira sprint list-workitems --board 686 --sprint 2354 --jql 'component = "Chap Modeling Platform"' --csv
```

`acli jira sprint update` is a full replace: `--name` and `--state` are required
even when you only mean to change the dates.

## Team members

**Assign by email, never by account ID.** `--assignee` claims to accept an
account ID, but when given one `acli` reports `SUCCESS` and **silently
unassigns the work item instead** — including wiping an existing assignee.
Both ID formats fail this way. Always verify after assigning.

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

Note two account-ID formats coexist: `712020:`-prefixed UUIDs (Ivar, Knut,
Edvin, Boris) and bare hex (Morten, Eirik, Abyot). Both are valid.

The CSV output renders assignee as an email, which is hidden for everyone but
yourself, so it shows blank. Use `--json` and read `assignee.displayName`.

## Getting Help

- General help: `acli --help`
- Jira commands: `acli jira --help`
- Workitem commands: `acli jira workitem --help`
- Specific command help: `acli jira workitem search --help`

## Additional Resources

- ACLI Documentation: https://developer.atlassian.com/cloud/acli/
- JQL Reference: https://support.atlassian.com/jira-software-cloud/docs/use-advanced-search-with-jira-query-language-jql/