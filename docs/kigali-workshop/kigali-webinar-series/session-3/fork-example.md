# 2. Implement your own model from a minimalist example
By the end of this guide, you will have a minimalist model running at your laptop in isolation and through chap, and have its code in your own GitHub repository.


Please consult the github guide of the previous session if you get unsure on how to fork and clone.

## Choose Your Language

We provide two minimalist example repositories demonstrating Chap-compatible models. 
They both implement the same simple linear regression model that predicts disease cases from rainfall and temperature.

You should choose one based on your preferred programming language (either R or Python):

| Language | Repository | Environment Manager |
|----------|-----------|---------------------|
| Python | [minimalist_example_uv](https://github.com/dhis2-chap/minimalist_example_uv) | uv |
| R | [minimalist_example_r](https://github.com/dhis2-chap/minimalist_example_r) | renv |

 

## Fork the Repository

1. Go to your chosen repository on GitHub:
   - **Python:** [github.com/dhis2-chap/minimalist_example_uv](https://github.com/dhis2-chap/minimalist_example_uv)
   - **R:** [github.com/dhis2-chap/minimalist_example_r](https://github.com/dhis2-chap/minimalist_example_r)

2. Click the **Fork** button in the top-right corner

3. Select your GitHub account as the destination

4. Keep the default settings and click **Create fork**

## Clone Your Fork

Clone the repository to your local machine. Replace `YOUR-USERNAME` with your GitHub username:

**Python:**
```bash
git clone https://github.com/YOUR-USERNAME/minimalist_example_uv.git
cd minimalist_example_uv
```

**R:**
```bash
git clone https://github.com/YOUR-USERNAME/minimalist_example_r.git
cd minimalist_example_r
```

## Next Steps

After cloning your fork, open the **README.md** in the repository or on Github. Go through the guide in the Readme. It contains instructions for:

- Running the model in isolated mode
- Making model alterations
- Running the model through CHAP

After following the README, and making changes to the code, you should commit and push changes to your fork on Github.

## Exercise

### Fork and clone the example

1. Fork one of the example repositories (Python or R)
2. Clone your fork to your local machine
3. Navigate to the repository directory
4. Follow the README in the repository. It guides you through running the model in isolated mode, how to make changes, and how to run the model through chap.
5. Make sure you commit and push any changes you make to your fork on Github.

**Verification:**

- You have a forked repository under your GitHub account
- You can see the README.md file in the repository
- You have been able to run the model in isolated mode
- You have been able to run your model through chap (using the chap evaluate command) and you have gotten a report.pdf with some results.
- You have been able to make changes, and you have pushed the changes to your fork on Github

