# plot-dataset Reference

`chap plot-dataset` plots the columns of a dataset CSV over time, one panel per location, so you can look at the data before evaluating a model on it.

```console
chap plot-dataset ./data.csv                                 # opens the plot in a browser
chap plot-dataset ./data.csv --no-browser                    # prints a local URL instead
chap plot-dataset ./data.csv --output-file ./data-plot.html  # saves the plot
```

By default the plot opens in your web browser. With `--no-browser`, it is served on a local address such as `http://127.0.0.1:53817/`, which is printed so you can open it yourself, for example on a machine without a browser through an SSH tunnel. The address works until you stop the command with Ctrl-C. `--output-file` saves the plot instead, as `.html`, `.png`, `.svg` or `.pdf`.

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `DATA-FILENAME` | The dataset CSV to plot | Required |
| `--plot-name` | Type of plot | `standardized-feature-plot` |
| `--output-file` | Save the plot to this file instead of showing it | None |
| `--browser` / `--no-browser` | Open the plot in a web browser, or print a local URL for it | `--browser` |
