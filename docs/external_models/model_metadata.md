# Describing your model with metadata

The `meta_data` block in your `MLproject` file controls how your model is displayed in the Chap Modeling App and catalogued in the Chap database (the name shown to users, author, organization logo, citation, etc.). All fields are optional — defaults are used when a field is omitted — but filling them in makes your model easier to recognize and trust.

## Supported fields

| Field | Type | Default | Purpose |
|---|---|---|---|
| `display_name` | string | `"No Display Name yet"` | Human-readable name shown in the UI |
| `description` | string | `"No Description yet"` | One- or two-paragraph summary of what the model does |
| `author` | string | `"Unknown Author"` | Person or team credited as the model author |
| `author_note` | string | `"No Author note yet"` | Free-form note from the author (caveats, intended use, etc.) |
| `author_assessed_status` | enum | `red` | Author's own maturity rating: `gray`, `red`, `orange`, `yellow`, `green` (see below) |
| `organization` | string | – | Organization the author belongs to |
| `organization_logo_url` | URL | – | Public URL to an image used as the organization logo in the UI |
| `contact_email` | string | – | Contact address for questions about the model |
| `citation_info` | string | – | How users should cite the model |
| `documentation_url` | URL | – | Link to external documentation for the model |

### `author_assessed_status` values

The author's own assessment of how mature the model is. This is shown in the UI as a colored badge.

| Value | Meaning |
|---|---|
| `gray` | Not intended for use, deprecated, or legacy-only |
| `red` | Highly experimental prototype, not validated |
| `orange` | Shows promise on limited data; needs manual configuration and careful evaluation |
| `yellow` | Ready for more rigorous testing |
| `green` | Validated and ready for use |

## Example

```yaml
name: my_model

uv_env: pyproject.toml

entry_points:
  train:
    parameters:
      train_data: str
      model: str
    command: "python train.py {train_data} {model}"
  predict:
    parameters:
      historic_data: str
      future_data: str
      model: str
      out_file: str
    command: "python predict.py {model} {historic_data} {future_data} {out_file}"

meta_data:
  display_name: "Monthly Deep Auto Regressive"
  description: >
    Experimental deep learning model based on an RNN architecture,
    focusing on predictions from auto-regressive time series data.
  author: "Knut Rand"
  author_assessed_status: orange
  organization: "HISP Centre, University of Oslo"
  organization_logo_url: "https://landportal.org/sites/default/files/2024-03/university_of_oslo_logo.png"
  contact_email: "knut.rand@dhis2.org"
  citation_info: >
    Rand, Knut. 2025. "Monthly Deep Auto Regressive model".
    HISP Centre, University of Oslo.
  documentation_url: "https://dhis2-chap.github.io/chap-core/external_models/overview_of_supported_models.html"
```

For full real-world MLproject files, see `example_data/model_templates/ar_monthly.yaml` and `example_data/model_templates/chap_ewars_monthly.yaml`.
