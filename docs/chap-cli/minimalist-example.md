# Run a simple evaluation

At this stage, you have installed Chap Core and want to verify that Chap can fetch a model, prepare data, and run a backtest end-to-end. You can do this with a single `chap eval` command.

## One-liner

Run an evaluation against the minimalist example model, using an example dataset hosted on GitHub. Pass `--plot` to generate an HTML visualization alongside the NetCDF output:

```bash
chap eval \
    --model-name https://github.com/dhis2-chap/minimalist_example_uv \
    --dataset-csv https://raw.githubusercontent.com/dhis2-chap/chap-core/master/example_data/laos_subset.csv \
    --output-file /tmp/chap/temp/eval.nc \
    --backtest-params.n-splits 2 \
    --backtest-params.n-periods 1 \
    --plot
```

## Verification

If the command completes and writes `/tmp/chap/temp/eval.nc` and `/tmp/chap/temp/eval.html`, your Chap installation is working: it can resolve a GitHub-hosted model, set up its environment, run a rolling-origin backtest on a remote dataset, and plot the results.

The next step is to integrate your own model into Chap.
