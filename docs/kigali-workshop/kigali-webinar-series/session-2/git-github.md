# 2. Git and GitHub
By the end of this guide, you will have your own account at GitHub, where you can conveniently access and share code with the community.   
## Why?

When working with code, you need to:

- **Access code others have shared** — download projects, examples, and libraries
- **Share your own code** — let others use, review, or collaborate on your work
- **Keep your code safe** — store it somewhere that won't disappear if your laptop breaks

GitHub is a website for sharing code. Git is a tool for downloading and uploading code to GitHub.

## What is GitHub?

GitHub is a **website for hosting and sharing code**. Think of it as Google Drive for code, but with features designed specifically for programmers.

Key concepts:

- **Repository (repo)**: A project folder containing code and files, hosted on GitHub. Each repo belongs to a user account or an organization.
- **Fork**: A copy of a repository on Github. If you fork a repository, you get your own copy of a repository on Github. You can modify it without affecting the original repository.

## Getting Started with GitHub

### Creating an Account

If you don't already have a GitHub account:

1. Go to [github.com](https://github.com)
2. Click "Sign up"
3. Follow the registration steps
4. Verify your email address

If you already have an account, just sign in.

### Browsing a Repository

You can explore any public repository without logging in:

1. Go to a repository URL, for example: [github.com/dhis2-chap/chap-workshop-python](https://github.com/dhis2-chap/chap-workshop-python)
2. You'll see:
   - **File list**: All the files and folders in the project
   - **README**: A description of the project (displayed at the bottom)
   - **Code button**: For downloading or cloning the code
   - **Fork button**: For creating your own copy (top right)

Click on any file to view its contents. Click on folders to navigate into them.

### Forking a Repository

Forking creates your own copy of a repository that you can modify:

1. Navigate to the repository you want to fork
2. Click the **"Fork"** button in the top right
3. Select your account as the destination
4. You now have your own copy at `github.com/YOUR-USERNAME/repo-name`

Your fork is completely independent — changes you make won't affect the original.

## What is Git?

Git is a **command-line tool** that works with GitHub. While GitHub is the website where code is stored, Git is the tool you use to:

- **Download** code from GitHub to your computer
- **Upload** your changes back to GitHub
- **Track changes** to your code over time (so you can see what changed and undo mistakes)

## Using Git

### Installation

**macOS:**

<details markdown="1">
<summary>Show command</summary>

```bash
brew install git
```

</details>

**Linux (Ubuntu/Debian) / WSL:**

<details markdown="1">
<summary>Show command</summary>

```bash
sudo apt update
sudo apt install git
```

</details>

### Initial Setup

Configure your identity (this labels your changes):

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### GitHub CLI (Authentication)

To push and pull code without entering passwords, install the GitHub CLI and authenticate:

**Installation:**

macOS:

<details markdown="1">
<summary>Show command</summary>

```bash
brew install gh
```

</details>

Linux (Ubuntu/Debian) / WSL:

<details markdown="1">
<summary>Show command</summary>

```bash
(type -p wget >/dev/null || (sudo apt update && sudo apt-get install wget -y)) \
&& sudo mkdir -p -m 755 /etc/apt/keyrings \
&& out=$(mktemp) && wget -nv -O$out https://cli.github.com/packages/githubcli-archive-keyring.gpg \
&& cat $out | sudo tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null \
&& sudo chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
&& echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
&& sudo apt update \
&& sudo apt install gh -y
```

</details>

**Authenticate:**

```bash
gh auth login
```

When prompted:

1. Select **GitHub.com**
2. Select **HTTPS**
3. Select **Yes** to authenticate with GitHub credentials
4. Select **Login with a web browser** and follow the prompts

Once authenticated, Git will use your GitHub credentials automatically.

### Cloning a Repository

"Cloning" means downloading a repository to your computer. You download a repo by running `git clone` followed by the repository link. For example, to clone the Chap Core repo:

```bash
git clone https://github.com/dhis2-chap/chap-core
```

This command creates a folder named `chap-core` in your current directory with all the repository files.

To enter the folder:

```bash
cd chap-core
```

### Making Changes

After cloning, you can edit files normally with any text editor. When you're ready to save your changes, you'll use a two-step process: **staging** and **committing**.

**1. Check what's changed:**

```bash
git status
```

This shows which files you've modified, added, or deleted.

**2. Stage your changes:**

```bash
git add .
```

Staging selects which changes you want to include in your next save. The `.` means "all changes", but you can also stage specific files with `git add filename.py`. Think of it as putting items in a box before shipping.

**3. Commit your changes:**

```bash
git commit -m "Describe what you changed"
```

Committing saves a snapshot of your staged changes with a message describing what you did. This creates a checkpoint you can return to later. The message helps you (and others) understand what changed and why.

### Pushing to GitHub

After committing, your changes are saved locally on your computer. To share them on GitHub, you need to **push**:

```bash
git push
```

This uploads your commits to GitHub, making them visible to others and backing them up online.

### Pulling Updates

If the repository has changed on GitHub (e.g., you made changes on another computer, or a collaborator pushed updates), you need to **pull** those changes to your local copy:

```bash
git pull
```

This downloads any new commits from GitHub and updates your local files.

## Exercise

### Part 1: GitHub (Web Interface)

**1. Create a GitHub account**

Go to [github.com](https://github.com) and sign up (if you haven't already).

**Verify**: You can log in to github.com

**2. Browse a repository**

Go to [github.com/dhis2-chap/chap-workshop-python](https://github.com/dhis2-chap/chap-workshop-python)

> **R users:** You can also use [github.com/dhis2-chap/chap-workshop-r](https://github.com/dhis2-chap/chap-workshop-r) instead.

**Verify**: You can see the list of files and the README at the bottom

**3. Fork the repository**

1. Click "Fork" in the top right
2. Select your account

**Verify**: You now have a copy at `github.com/YOUR-USERNAME/chap-workshop-python`

### Part 2: Git (Command Line)

**4. Check Git is installed**

```bash
git --version
```

**Verify**: You should see a version number like `git version 2.x.x`

**5. Configure your identity**

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

**Verify**: Run `git config --list` and confirm your name and email appear

**6. Install GitHub CLI and authenticate**

Install GitHub CLI:

macOS:

<details markdown="1">
<summary>Show command</summary>

```bash
brew install gh
```

</details>

Linux (Ubuntu/Debian) / WSL:

<details markdown="1">
<summary>Show command</summary>

```bash
(type -p wget >/dev/null || (sudo apt update && sudo apt-get install wget -y)) \
&& sudo mkdir -p -m 755 /etc/apt/keyrings \
&& out=$(mktemp) && wget -nv -O$out https://cli.github.com/packages/githubcli-archive-keyring.gpg \
&& cat $out | sudo tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null \
&& sudo chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
&& echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
&& sudo apt update \
&& sudo apt install gh -y
```

</details>

Then authenticate:

```bash
gh auth login
```

Select: **GitHub.com** → **HTTPS** → **Yes** → **Login with a web browser**

**Verify**: Run `gh auth status` and confirm you're logged in

**7. Clone your fork**

First, navigate to where you want the repository folder to be created. For example, to put it in your home directory:

```bash
cd ~
```

Now clone your fork. This creates a new folder named `chap-workshop-python` in your current location:

```bash
git clone https://github.com/YOUR-USERNAME/chap-workshop-python.git
cd chap-workshop-python
```

**Verify**: Run `ls` and you should see the repository files

**8. Check the remote**

```bash
git remote -v
```

**Verify**: You should see `origin` pointing to your GitHub fork

**9. Make a change**

To find where the repository is on your computer, run:

```bash
pwd
```

This shows the full path (e.g., `/home/username/chap-workshop-python`). Open this folder in your code editor (like VS Code), or navigate to it in your file explorer.

Open the `README.md` file in your editor, add a line (e.g., your name or a note), and save the file.

**10. Check the status**

```bash
git status
```

**Verify**: You should see `README.md` listed as modified (in red)

**11. Stage, commit, and push**

```bash
git add README.md
git status
```

**Verify**: `README.md` should now be listed as staged (in green)

```bash
git commit -m "Add my name to README"
git push
```

**Verify**: Visit your fork on GitHub and you should see your change in the README

If all verifications passed, you're ready for the next guide: [Virtual Environments](virtual-environments.md)

## Learn More

Want to dive deeper? Here are some helpful resources:

**GitHub:**

- [GitHub's "What is GitHub?" guide](https://docs.github.com/en/get-started/start-your-journey/about-github-and-git) — Official introduction to GitHub and how it works
- [GitHub Skills](https://skills.github.com/) — Free interactive courses to learn GitHub

**Git:**

- [Git Cheat Sheet (GitHub)](https://education.github.com/git-cheat-sheet-education.pdf) — Quick reference for common Git commands
- [Atlassian Git Tutorials](https://www.atlassian.com/git/tutorials) — In-depth explanations of Git concepts and commands
- [Pro Git Book](https://git-scm.com/book/en/v2) — Free comprehensive book on Git (if you want to understand everything)
