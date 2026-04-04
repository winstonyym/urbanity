# Contributing

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

## Types of Contributions

### Report Bugs

If you are reporting a bug, please include:

- Your operating system name and version.
- Any details about your local setup that might be helpful in troubleshooting.
- Detailed steps to reproduce the bug.

Open a bug report on [GitHub Issues](https://github.com/winstonyym/urbanity/issues).

### Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help wanted" is open to whoever wants to implement it.

### Implement Features

Look through the GitHub issues for features. Anything tagged with "enhancement" and "help wanted" is open to whoever wants to implement it.

### Write Documentation

You can never have enough documentation! Feel free to contribute to any part of the documentation, such as the official docs, docstrings, or even blog posts and articles.

### Submit Feedback

If you are proposing a feature:

- Explain in detail how it would work.
- Keep the scope as narrow as possible, to make it easier to implement.
- Remember that this is a volunteer-driven project, and contributions are welcome :)

## Get Started

Ready to contribute? Here's how to set up `urbanity` for local development.

1. Fork the repository on GitHub.
2. Clone your fork locally:

    ```bash
    git clone https://github.com/YOUR_USERNAME/urbanity.git
    cd urbanity
    ```

3. Create a virtual environment and install in editable mode:

    ```bash
    uv venv --python 3.11
    source .venv/bin/activate
    uv pip install -e ".[osm]"
    ```

4. Create a branch for your changes:

    ```bash
    git checkout -b name-of-your-bugfix-or-feature
    ```

5. Make your changes, then open a pull request.

## Pull Request Guidelines

Before you submit a pull request, please check that it meets these guidelines:

- [ ] The pull request includes tests if appropriate.
- [ ] If the pull request adds functionality, the docs are updated.
- [ ] The pull request works for all currently supported operating systems and Python versions.