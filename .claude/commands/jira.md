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

## Getting Help

- General help: `acli --help`
- Jira commands: `acli jira --help`
- Workitem commands: `acli jira workitem --help`
- Specific command help: `acli jira workitem search --help`

## Additional Resources

- ACLI Documentation: https://developer.atlassian.com/cloud/acli/
- JQL Reference: https://support.atlassian.com/jira-software-cloud/docs/use-advanced-search-with-jira-query-language-jql/