# File: `.github/scripts/check_prerelease_dependencies.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 36 |
| Imports | sys, tomllib |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Validates that stable releases don't depend on prerelease versions of dependencies.

**Mechanism:** Reads a pyproject.toml file, checks if the current version contains "rc" or "dev", and if not (stable release), scans all dependencies to ensure none have prerelease markers or allow-prereleases flags. Raises ValueError if violations are found.

**Significance:** Release validation guard that prevents accidentally publishing stable versions with unstable dependencies, ensuring production reliability and avoiding downstream version conflicts.
