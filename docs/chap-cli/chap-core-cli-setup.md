# 1. Installing Chap for model developers

In this guide, you'll install the Chap command-line tool. Once installed, you can run `chap eval` to test any model against real datasets — which you'll do in the next guide in this session.

**Reminder:** Windows users, use WSL (Windows Subsystem for Linux) as covered in [Prepare for installation](../external_models/prepare-for-installation.md).

## Installing Chap

Install Chap as a global tool using uv:

```bash
uv tool install chap-core --python 3.13
```

This installs the `chap` command-line tool globally, making it available from any directory.

## Exercise

### Verify your installation

Run the following command:

```bash
chap --help
```

You should see output listing available commands including `eval`, `plot-backtest`, and `export-metrics`.

**Verification:** If you see the help output with available commands, Chap is installed correctly. You're ready for the next guide: [Implement your own model from a minimalist example](../kigali-workshop/kigali-webinar-series/session-3/fork-example.md).
