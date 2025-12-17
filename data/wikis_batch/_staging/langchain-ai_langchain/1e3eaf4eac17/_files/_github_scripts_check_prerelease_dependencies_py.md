# File: `.github/scripts/check_prerelease_dependencies.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 36 |
| Imports | sys, tomllib |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Validates that package dependencies do not allow prerelease versions (rc, dev) unless the package itself is being released as a prerelease. This ensures stable releases only depend on stable dependencies.

**Mechanism:** The script accepts a pyproject.toml file path as a command-line argument. It parses the TOML file to extract the package version and checks if it contains "rc" or "dev" markers indicating a prerelease. If the package is not a prerelease, it iterates through all dependencies and validates that none have "rc" in their version strings or have the "allow-prereleases" flag set to true. If any dependency violates these rules, it raises a ValueError with a descriptive message.

**Significance:** Enforces release quality standards by preventing accidental stable releases that depend on unstable prerelease packages. This is critical for maintaining production reliability, as stable versions should never transitively pull in alpha/beta/rc dependencies that could introduce unexpected breaking changes or bugs. This check runs as part of the release validation workflow.
