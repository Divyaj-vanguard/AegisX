#!/bin/bash
# Stage all changes and new untracked files
git add -A

# Check if there are changes before committing
if ! git diff --cached --quiet; then
    git commit -m "Auto-commit on $(date '+%Y-%m-%d %H:%M:%S')"
    git push origin main
fi
