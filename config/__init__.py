"""
RAGForge configuration package.

Centralises all project settings so that every other module imports
configuration values from ``config.settings`` rather than hard-coding
them.  This keeps the codebase DRY and makes it trivial to change
thresholds, model names, or directory paths in one place.
"""
