#!/bin/bash
# import_wikis.sh - Import .mediawiki files from /wikis into MediaWiki
#
# This script scans the /wikis directory for .mediawiki files and imports them
# into the wiki using MediaWiki's maintenance/edit.php script.
#
# Directory structure expected:
#   /wikis/
#     repo_name_1/
#       Page_Name.mediawiki
#       Another_Page.mediawiki
#     repo_name_2/
#       ...
#
# Pages are created with titles: {repo_name}/{Page_Name}
# e.g., huggingface_text-generation-inference/Workflow_Model_Loading

set -euo pipefail

# Configuration
WIKIS_DIR="${WIKIS_DIR:-/wikis}"
# Store flag in mounted volume so it persists across container restarts
IMPORT_FLAG="${WIKIS_DIR}/.wikis_imported"
MW_USER="${MW_ADMIN_USER:-admin}"
FORCE_REIMPORT="${FORCE_REIMPORT:-false}"

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

echo "📥 Importing wiki pages from $WIKIS_DIR..."

# Counter for imported pages
imported=0
failed=0

# Find all .mediawiki files in subdirectories
while IFS= read -r -d '' file; do
    # Extract repo name and page name from path
    # e.g., /wikis/huggingface_text-generation-inference/Workflow_Model_Loading.mediawiki
    #       -> repo: huggingface_text-generation-inference
    #       -> page: Workflow_Model_Loading
    
    rel_path="${file#$WIKIS_DIR/}"
    repo_name=$(dirname "$rel_path")
    page_name=$(basename "$file" .mediawiki)
    
    # Skip domain_tag.txt and other non-wiki files
    if [[ "$page_name" == "domain_tag" ]]; then
        continue
    fi
    
    # Convert filename prefix to MediaWiki namespace
    # e.g., Principle_Token_Streaming -> Principle:repo_name/Token_Streaming
    # Supported prefixes: Principle, Workflow, Implementation, Artifact, Heuristic, Environment, Resource
    if [[ "$page_name" =~ ^(Principle|Workflow|Implementation|Artifact|Heuristic|Environment|Resource)_(.+)$ ]]; then
        namespace="${BASH_REMATCH[1]}"
        name="${BASH_REMATCH[2]}"
        # Format: Namespace:repo_name/Page_Name
        page_title="${namespace}:${repo_name}/${name}"
    else
        # No recognized prefix, use default format
        page_title="${repo_name}/${page_name}"
    fi
    
    echo "  📄 Importing: $page_title"
    
    # Import the page using MediaWiki's edit.php maintenance script
    # Note: edit.php may return non-zero if "no change was made", which is fine
    output=$(php /var/www/html/maintenance/run.php edit.php \
        --user "$MW_USER" \
        --summary "Auto-imported from $rel_path" \
        "$page_title" < "$file" 2>&1) || true
    
    if echo "$output" | grep -qE "(Saving.*done|no change was made)"; then
        imported=$((imported + 1))
    else
        echo "    ⚠️  Failed to import: $page_title"
        echo "    $output"
        failed=$((failed + 1))
    fi
    
done < <(find "$WIKIS_DIR" -type f -name "*.mediawiki" -print0)

# Create import flag file with timestamp
echo "Imported at: $(date -Iseconds)" > "$IMPORT_FLAG"
echo "Pages imported: $imported" >> "$IMPORT_FLAG"

echo ""
echo "✅ Import complete: $imported pages imported, $failed failed"

