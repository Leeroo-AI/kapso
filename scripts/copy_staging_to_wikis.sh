#!/bin/bash
# Copy contents from each staging repo's subfolders into the main wikis folders.
#
# Structure:
#   data/wikis/_staging/<Repo>/<hash>/{environments,heuristics,implementations,principles,workflows}/
#   -> data/wikis/{environments,heuristics,implementations,principles,workflows}/

set -euo pipefail

WIKIS_DIR="/home/ubuntu/kapso/data/wikis"
STAGING_DIR="$WIKIS_DIR/_staging"
FOLDERS=("environments" "heuristics" "implementations" "principles" "workflows")

copied=0
skipped=0

# Loop over each repo in staging
for repo_dir in "$STAGING_DIR"/*/; do
    # Each repo has a hash subdirectory
    for hash_dir in "$repo_dir"/*/; do
        [ -d "$hash_dir" ] || continue

        for folder in "${FOLDERS[@]}"; do
            src="$hash_dir$folder"
            dst="$WIKIS_DIR/$folder"

            # Skip if source folder doesn't exist or is empty
            if [ ! -d "$src" ]; then
                continue
            fi

            # Copy all files from source to destination
            file_count=$(ls -1 "$src" 2>/dev/null | wc -l)
            if [ "$file_count" -gt 0 ]; then
                cp -n "$src"/* "$dst"/
                copied=$((copied + file_count))
                echo "Copied $file_count files from $src -> $dst"
            fi
        done
    done
done

echo ""
echo "Done. Copied $copied files total."
