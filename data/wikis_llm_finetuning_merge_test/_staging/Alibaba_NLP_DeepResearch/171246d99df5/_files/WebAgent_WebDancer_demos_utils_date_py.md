# File: `WebAgent/WebDancer/demos/utils/date.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 71 |
| Functions | `get_date_now`, `get_date_rand`, `str2date`, `date2str` |
| Imports | datetime, random |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides date manipulation utilities for time-aware prompting in WebDancer.

**Mechanism:** Functions operate on date as [year, month, day, weekday] lists. Key functions: (1) `get_date_now()` - returns current Beijing time (UTC+8) as date list, (2) `get_date_rand()` - generates random date within specified day range around now, (3) `str2date()` - parses date string (default YYYY-MM-DD format) to date list, (4) `date2str()` - formats date list to string with configurable separator, optional weekday name in English or Chinese. Contains weekday name mappings for 'en' (Monday-Sunday) and 'zh' (Chinese characters).

**Significance:** Utility component enabling WebDancer's time awareness. Used in system prompts to inform the agent of current date, helping it provide contextually relevant and time-sensitive information in responses.
