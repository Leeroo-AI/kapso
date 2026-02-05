#!/bin/bash
# import_wikis.sh - Import wiki files from /wikis into MediaWiki
#
# This script scans the /wikis directory for wiki files and imports them
# into the wiki using MediaWiki's maintenance/edit.php script.
#
# Supported file formats (in order of preference):
#   1. .mediawiki files
#   2. .md files (fallback if no .mediawiki files found)
#
# Directory structure expected:
#   /wikis/
#     repo_name_1/
#       Page_Name.mediawiki (or .md)
#       Another_Page.mediawiki (or .md)
#     repo_name_2/
#       ...
#
# Pages are created with titles: {repo_name}/{Page_Name}
# e.g., huggingface_text-generation-inference/Workflow_Model_Loading

set -eu
# Note: pipefail removed to avoid SIGPIPE errors when output is redirected

# Configuration
WIKIS_DIR="${WIKIS_DIR:-/wikis}"
# Store flag in mounted volume so it persists across container restarts
IMPORT_FLAG="${WIKIS_DIR}/.wikis_imported"
MW_USER="${MW_ADMIN_USER:-admin}"
FORCE_REIMPORT="${FORCE_REIMPORT:-false}"

# Folders to skip during import
SKIP_FOLDERS="_staging|_reports|_files"

# Map folder names to MediaWiki namespaces
# Folders are lowercase plural, namespaces are capitalized singular
declare -A FOLDER_TO_NAMESPACE=(
    ["workflows"]="Workflow"
    ["principles"]="Principle"
    ["implementations"]="Implementation"
    ["artifacts"]="Artifact"
    ["heuristics"]="Heuristic"
    ["environments"]="Environment"
    ["resources"]="Resource"
)

# Check if wikis directory exists
if [ ! -d "$WIKIS_DIR" ]; then
    echo "ℹ️  No wikis directory found at $WIKIS_DIR, skipping import"
    exit 0
fi

# Check if already imported (unless force reimport)
if [ -f "$IMPORT_FLAG" ] && [ "$FORCE_REIMPORT" != "true" ]; then
    echo "✓ Wiki pages already imported (use FORCE_REIMPORT=true to reimport)"
    exit 0
fi

# Determine which file extension to use
# Prefer .mediawiki, fall back to .md if no .mediawiki files found
mediawiki_count=$(find "$WIKIS_DIR" -type f -name "*.mediawiki" 2>/dev/null | wc -l)
md_count=$(find "$WIKIS_DIR" -type f -name "*.md" 2>/dev/null | wc -l)

if [ "$mediawiki_count" -gt 0 ]; then
    FILE_EXT="mediawiki"
    echo "📥 Importing wiki pages from $WIKIS_DIR (found $mediawiki_count .mediawiki files)..."
elif [ "$md_count" -gt 0 ]; then
    FILE_EXT="md"
    echo "📥 Importing wiki pages from $WIKIS_DIR (found $md_count .md files, using as fallback)..."
else
    echo "ℹ️  No .mediawiki or .md files found in $WIKIS_DIR, skipping import"
    exit 0
fi

# Counter for imported pages
imported=0
failed=0
skipped=0

# Find all wiki files in subdirectories
while IFS= read -r -d '' file; do
    # Extract relative path and folder name
    rel_path="${file#$WIKIS_DIR/}"
    folder_name=$(echo "$rel_path" | cut -d'/' -f1)
    
    # Skip files in staging/reports/files folders
    if echo "$rel_path" | grep -qE "($SKIP_FOLDERS)"; then
        skipped=$((skipped + 1))
        continue
    fi
    
    # Skip files starting with underscore
    filename=$(basename "$file")
    if [[ "$filename" == _* ]]; then
        skipped=$((skipped + 1))
        continue
    fi
    
    # Extract page name (remove extension)
    page_name=$(basename "$file" ".$FILE_EXT")
    
    # Skip domain_tag and other non-wiki files
    if [[ "$page_name" == "domain_tag" ]] || [[ "$page_name" == "README" ]]; then
        skipped=$((skipped + 1))
        continue
    fi
    
    # Build page title using folder-to-namespace mapping
    # e.g., workflows/Unslothai_Unsloth_QLoRA_Finetuning.md -> Workflow:Unslothai_Unsloth_QLoRA_Finetuning
    namespace="${FOLDER_TO_NAMESPACE[$folder_name]:-}"
    
    if [ -n "$namespace" ]; then
        # Known folder -> use namespace with colon
        page_title="${namespace}:${page_name}"
    else
        # Unknown folder -> use folder/filename format (main namespace)
        page_title="${folder_name}/${page_name}"
    fi
    
    echo "  📄 Importing: $page_title"
    
    # Read file content and strip redundant H1 heading if present
    # The page title already comes from the namespace, so first-line headings are redundant
    # Strips lines starting with: "# " (Markdown H1) or "= " (MediaWiki H1)
    content=$(cat "$file")
    first_line=$(echo "$content" | head -n1)
    if [[ "$first_line" =~ ^#[[:space:]] ]] || [[ "$first_line" =~ ^=[[:space:]] ]]; then
        # Strip the first heading line and any following blank line
        content=$(echo "$content" | tail -n +2 | sed '/./,$!d')
    fi
    
    # Transform source links to clickable external links
    # Pattern: [[source::Type|DisplayText|URL]] -> [URL DisplayText]
    # Example: [[source::Repo|Unsloth|https://github.com/...]] -> [https://github.com/... Unsloth]
    content=$(echo "$content" | sed -E 's/\[\[source::[^|]+\|([^|]+)\|(https?:[^]]+)\]\]/[\2 \1]/g')

    # Add category and Cargo template for queryability
    if [ -n "$namespace" ]; then
        category="${namespace}s"  # Pluralize: Workflow -> Workflows
        # Add PageInfo template for Cargo queries and category
        content="{{PageInfo|type=${namespace}|title=${page_name}}}
${content}

[[Category:${category}]]"
    fi

    # Import the page using MediaWiki's edit.php maintenance script
    # Note: edit.php may return non-zero if "no change was made", which is fine
    output=$(echo "$content" | php /var/www/html/maintenance/run.php edit.php \
        --user "$MW_USER" \
        --summary "Auto-imported from $rel_path" \
        "$page_title" 2>&1) || true
    
    if echo "$output" | grep -qE "(Saving.*done|no change was made)"; then
        imported=$((imported + 1))
    else
        echo "    ⚠️  Failed to import: $page_title"
        echo "    $output"
        failed=$((failed + 1))
    fi
    
done < <(find "$WIKIS_DIR" -type f -name "*.$FILE_EXT" -print0)

# Create import flag file with timestamp
echo "Imported at: $(date -Iseconds)" > "$IMPORT_FLAG"
echo "Pages imported: $imported" >> "$IMPORT_FLAG"
echo "File format: $FILE_EXT" >> "$IMPORT_FLAG"

echo ""
echo "✅ Import complete: $imported imported, $skipped skipped, $failed failed"

# Recreate Cargo table to index all imported pages
# Always rebuild to ensure table schema is correct (--replacement creates fresh table with all internal columns)
echo "🔄 Rebuilding Cargo PageInfo table..."
php /var/www/html/extensions/Cargo/maintenance/cargoRecreateData.php --table PageInfo --replacement 2>/dev/null || true
echo "✓ Cargo table rebuilt"

