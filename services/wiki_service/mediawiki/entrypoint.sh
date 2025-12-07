#!/bin/bash
# MediaWiki entrypoint script for local development
# This script runs before the Apache web server starts
# It initializes the database and creates necessary users

set -euo pipefail

# Wait for MariaDB to be ready before proceeding
echo "⏳ Waiting for database..."
until mysql -h "$MW_DB_HOST" -u "$MW_DB_USER" -p"$MW_DB_PASSWORD" -e "SELECT 1" >/dev/null 2>&1; do
  sleep 2
done
echo "✓ Database is ready"

# Check if LocalSettings.php exists (wiki already initialized)
if [ ! -f /var/www/html/LocalSettings.php ]; then
    echo "🚀 Initializing MediaWiki..."
    
    # Run MediaWiki installation script
    # This creates the database tables and generates LocalSettings.php
    php maintenance/run.php install.php \
        --dbname="$MW_DB_NAME" \
        --dbserver="$MW_DB_HOST" \
        --dbuser="$MW_DB_USER" \
        --dbpass="$MW_DB_PASSWORD" \
        --lang="$MW_LANG" \
        --pass="$MW_ADMIN_PASS" \
        --scriptpath="" \
        --server="$MW_SITE_SERVER" \
        "$MW_SITENAME" \
        "$MW_ADMIN_USER"
    
    # Append extension configuration to LocalSettings.php
    cat >> /var/www/html/LocalSettings.php <<'EOF'

# ============================================
# Extensions (loaded by entrypoint)
# ============================================
wfLoadExtension('SemanticMediaWiki');
wfLoadExtension('PageForms');
wfLoadExtension('Cargo');
wfLoadExtension('VisualEditor');
wfLoadExtension('SyntaxHighlight_GeSHi');
wfLoadExtension('Math');
wfLoadExtension('Mermaid');
wfLoadExtension('DynamicPageList3');
wfLoadExtension('Network');

# ============================================
# SyntaxHighlight configuration
# ============================================
$wgPygmentizePath = '/usr/bin/pygmentize';

# ============================================
# Math extension configuration
# ============================================
$wgDefaultUserOptions['math'] = 'native';
$wgMathValidModes = ['native', 'mathml', 'mathjax', 'source'];
$wgMathEnableWikibaseDataType = true;

# ============================================
# Mermaid diagrams configuration
# ============================================
$wgMermaidDefaultTheme = 'default';
$wgMermaidDefaultOptions = [
    'theme' => 'default',
    'themeVariables' => [
        'primaryColor' => '#f8f9fa',
        'primaryTextColor' => '#000',
        'primaryBorderColor' => '#a2a9b1',
        'lineColor' => '#000',
        'secondaryColor' => '#eaecf0',
        'tertiaryColor' => '#fff'
    ]
];

# ============================================
# VisualEditor configuration
# ============================================
$wgDefaultUserOptions['visualeditor-enable'] = 1;
$wgVirtualRestConfig['modules']['parsoid'] = [
    'url' => 'http://localhost:8142',
    'forwardCookies' => true,
];

# ============================================
# File uploads
# ============================================
$wgEnableUploads = true;
$wgFileExtensions = [ 'png', 'gif', 'jpg', 'jpeg', 'pdf', 'svg', 'webp' ];
$wgMaxUploadSize = 1024 * 1024 * 50; # 50MB

# ============================================
# API configuration
# ============================================
$wgEnableAPI = true;
$wgEnableWriteAPI = true;

# ============================================
# Semantic MediaWiki
# ============================================
enableSemantics();

# ============================================
# Appearance (logo)
# ============================================
$wgLogo = "/leeroo-logo.png";
$wgLogos = [
    '1x' => "/leeroo-logo.png",
    '2x' => "/leeroo-logo.png",
    'icon' => "/leeroo-logo.png"
];
$wgFavicon = "/leeroo-logo.png";

# ============================================
# Suppress PHP deprecation warnings
# (MediaWiki 1.43 + PHP 8.x compatibility)
# ============================================
error_reporting(E_ALL & ~E_DEPRECATED & ~E_STRICT);

# ============================================
# Custom Namespaces for Knowledge Organization
# ============================================
# Define namespace IDs (must be even numbers, starting from 3000)
# Note: "Concept" conflicts with SMW's existing namespace (ID 108), so we use "Principle"
define("NS_PRINCIPLE", 3000);
define("NS_PRINCIPLE_TALK", 3001);
define("NS_WORKFLOW", 3002);
define("NS_WORKFLOW_TALK", 3003);
define("NS_IMPLEMENTATION", 3004);
define("NS_IMPLEMENTATION_TALK", 3005);
define("NS_ARTIFACT", 3006);
define("NS_ARTIFACT_TALK", 3007);
define("NS_HEURISTIC", 3008);
define("NS_HEURISTIC_TALK", 3009);
define("NS_ENVIRONMENT", 3010);
define("NS_ENVIRONMENT_TALK", 3011);
define("NS_RESOURCE", 3012);
define("NS_RESOURCE_TALK", 3013);

# Register namespace names
$wgExtraNamespaces[NS_PRINCIPLE] = "Principle";
$wgExtraNamespaces[NS_PRINCIPLE_TALK] = "Principle_talk";
$wgExtraNamespaces[NS_WORKFLOW] = "Workflow";
$wgExtraNamespaces[NS_WORKFLOW_TALK] = "Workflow_talk";
$wgExtraNamespaces[NS_IMPLEMENTATION] = "Implementation";
$wgExtraNamespaces[NS_IMPLEMENTATION_TALK] = "Implementation_talk";
$wgExtraNamespaces[NS_ARTIFACT] = "Artifact";
$wgExtraNamespaces[NS_ARTIFACT_TALK] = "Artifact_talk";
$wgExtraNamespaces[NS_HEURISTIC] = "Heuristic";
$wgExtraNamespaces[NS_HEURISTIC_TALK] = "Heuristic_talk";
$wgExtraNamespaces[NS_ENVIRONMENT] = "Environment";
$wgExtraNamespaces[NS_ENVIRONMENT_TALK] = "Environment_talk";
$wgExtraNamespaces[NS_RESOURCE] = "Resource";
$wgExtraNamespaces[NS_RESOURCE_TALK] = "Resource_talk";

# Enable subpages in custom namespaces
$wgNamespacesWithSubpages[NS_PRINCIPLE] = true;
$wgNamespacesWithSubpages[NS_WORKFLOW] = true;
$wgNamespacesWithSubpages[NS_IMPLEMENTATION] = true;
$wgNamespacesWithSubpages[NS_ARTIFACT] = true;
$wgNamespacesWithSubpages[NS_HEURISTIC] = true;
$wgNamespacesWithSubpages[NS_ENVIRONMENT] = true;
$wgNamespacesWithSubpages[NS_RESOURCE] = true;

# Enable Semantic MediaWiki in custom namespaces
$smwgNamespacesWithSemanticLinks[NS_PRINCIPLE] = true;
$smwgNamespacesWithSemanticLinks[NS_WORKFLOW] = true;
$smwgNamespacesWithSemanticLinks[NS_IMPLEMENTATION] = true;
$smwgNamespacesWithSemanticLinks[NS_ARTIFACT] = true;
$smwgNamespacesWithSemanticLinks[NS_HEURISTIC] = true;
$smwgNamespacesWithSemanticLinks[NS_ENVIRONMENT] = true;
$smwgNamespacesWithSemanticLinks[NS_RESOURCE] = true;

# Make custom namespaces searchable by default
$wgNamespacesToBeSearchedDefault[NS_PRINCIPLE] = true;
$wgNamespacesToBeSearchedDefault[NS_WORKFLOW] = true;
$wgNamespacesToBeSearchedDefault[NS_IMPLEMENTATION] = true;
$wgNamespacesToBeSearchedDefault[NS_ARTIFACT] = true;
$wgNamespacesToBeSearchedDefault[NS_HEURISTIC] = true;
$wgNamespacesToBeSearchedDefault[NS_ENVIRONMENT] = true;
$wgNamespacesToBeSearchedDefault[NS_RESOURCE] = true;

# ============================================
# Network Extension (Interactive Page Graphs)
# ============================================
# Allows {{#network:PageName}} to show interactive graph of page links
# Double-click nodes to navigate, hold to expand connections
$wgPageNetworkExcludeTalkPages = true;
$wgPageNetworkEnableDisplayTitle = true;
$wgPageNetworkLabelMaxLength = 50;
# Exclude system namespaces: User(2), Project(4), MediaWiki(8), Template(10), Help(12)
$wgPageNetworkExcludedNamespaces = [2, 4, 8, 10, 12];

# Vis.js options - using dot shapes (external links filtered via JS)
$wgPageNetworkOptions = [
    'nodes' => [
        'shape' => 'dot',
        'size' => 18,
        'font' => [
            'size' => 13,
            'face' => 'Arial, sans-serif',
            'strokeWidth' => 3,
            'strokeColor' => '#ffffff',
        ],
        'borderWidth' => 2,
        'shadow' => true,
    ],
    'groups' => [
        'bluelink' => [
            'shape' => 'dot',
            'color' => ['background' => '#2196F3', 'border' => '#1565C0'],
        ],
        'redlink' => [
            'shape' => 'dot',
            'size' => 14,
            'color' => ['background' => '#ffcdd2', 'border' => '#ef9a9a'],
            'font' => ['color' => '#c62828'],
        ],
        'externallink' => [
            'hidden' => true,
            'physics' => false,
        ],
    ],
    'edges' => [
        'width' => 1.5,
        'color' => ['color' => '#90CAF9', 'highlight' => '#2196F3'],
        'smooth' => ['type' => 'continuous'],
        'arrows' => ['to' => ['enabled' => true, 'scaleFactor' => 0.5]],
    ],
    'physics' => [
        'barnesHut' => [
            'gravitationalConstant' => -4000,
            'centralGravity' => 0.3,
            'springLength' => 120,
        ],
        'stabilization' => ['iterations' => 150],
    ],
    'interaction' => [
        'hover' => true,
        'tooltipDelay' => 100,
        'navigationButtons' => true,
    ],
];

# ============================================
# Auto-append Knowledge Graph to all pages
# ============================================
# Automatically show the network graph on pages in custom namespaces
$wgHooks['ArticleViewFooter'][] = function ( $article ) {
    $title = $article->getTitle();
    $ns = $title->getNamespace();
    
    // Only add to custom knowledge namespaces (3000-3012)
    $knowledgeNamespaces = [
        3000, // Principle
        3002, // Workflow
        3004, // Implementation
        3006, // Artifact
        3008, // Heuristic
        3010, // Environment
        3012, // Resource
    ];
    
    if ( in_array( $ns, $knowledgeNamespaces ) ) {
        $out = $article->getContext()->getOutput();
        $fullPageName = $title->getPrefixedDBkey();
        
        // Add the network graph section
        $graphWikitext = "
== Page Connections ==
<div style=\"font-size:0.9em; color:#666; margin-bottom:0.5em;\">'''Double-click''' a node to navigate. '''Hold''' to expand connections.</div>
{{NetworkLegend}}
{{#network:{$fullPageName}|depth=2}}
";
        $out->addWikiTextAsContent( $graphWikitext );
    }
    return true;
};
EOF

    echo "✓ MediaWiki initialized"
    
    # Setup Semantic MediaWiki data store
    echo "🔄 Setting up Semantic MediaWiki..."
    php extensions/SemanticMediaWiki/maintenance/setupStore.php --skip-optimize --quiet
    echo "✓ Semantic MediaWiki ready"
    
    # Run database update to create tables for all extensions (Math, Cargo, etc.)
    echo "🔄 Running database update for extensions..."
    php maintenance/run.php update.php --quick 2>/dev/null || true
    echo "✓ Extension tables created"
    
    # Create Main Page with DynamicPageList index
    echo "📄 Creating Main Page..."
    cat > /tmp/main_page.txt <<'MAINPAGE'
== Welcome to {{SITENAME}} ==

This wiki contains structured knowledge organized by type.

== Browse by Category ==

{| class="wikitable" style="width:100%"
|-
! Category !! Description !! Browse
|-
| '''Resources''' || '''Main entry points for each repository''' || [[Special:AllPages/Resource:|Browse All]]
|-
| '''Workflows''' || Step-by-step processes and procedures || [[Special:AllPages/Workflow:|Browse All]]
|-
| '''Principles''' || Core ideas and foundational knowledge || [[Special:AllPages/Principle:|Browse All]]
|-
| '''Implementations''' || Code-level details and modules || [[Special:AllPages/Implementation:|Browse All]]
|-
| '''Artifacts''' || Data structures and objects || [[Special:AllPages/Artifact:|Browse All]]
|-
| '''Heuristics''' || Best practices and guidelines || [[Special:AllPages/Heuristic:|Browse All]]
|-
| '''Environments''' || Setup and configuration guides || [[Special:AllPages/Environment:|Browse All]]
|}

== Recent Pages ==

=== Resources (Entry Points) ===
<DPL>
namespace=Resource
ordermethod=title
count=10
</DPL>

=== Workflows ===
<DPL>
namespace=Workflow
ordermethod=title
count=10
</DPL>

=== Principles ===
<DPL>
namespace=Principle
ordermethod=title
count=10
</DPL>

=== Implementations ===
<DPL>
namespace=Implementation
ordermethod=title
count=10
</DPL>

=== Artifacts ===
<DPL>
namespace=Artifact
ordermethod=title
count=10
</DPL>

=== Heuristics ===
<DPL>
namespace=Heuristic
ordermethod=title
count=10
</DPL>

=== Environments ===
<DPL>
namespace=Environment
ordermethod=title
count=10
</DPL>

----

''Use the category table above to browse all pages, or search using the search box.''
MAINPAGE
    
    php maintenance/run.php edit.php \
        --user "$MW_ADMIN_USER" \
        --summary "Auto-created: Main page with DPL index" \
        "Main Page" < /tmp/main_page.txt
    rm /tmp/main_page.txt
    echo "✓ Main Page created"
    
    # Create NetworkLegend template for graph visualization
    echo "📄 Creating Network Legend template..."
    cat > /tmp/network_legend.txt <<'LEGENDTEMPLATE'
<div class="network-legend" style="display:flex; flex-wrap:wrap; gap:12px; padding:12px; background:#fff; border:1px solid #a2a9b1; border-radius:6px; margin-bottom:1em; font-size:0.9em;">
<div style="display:flex; align-items:center; gap:6px;"><div style="width:14px; height:14px; border-radius:50%; background:#4CAF50; border:2px solid #2E7D32;"></div> Workflow</div>
<div style="display:flex; align-items:center; gap:6px;"><div style="width:14px; height:14px; border-radius:50%; background:#2196F3; border:2px solid #1565C0;"></div> Principle</div>
<div style="display:flex; align-items:center; gap:6px;"><div style="width:14px; height:14px; border-radius:50%; background:#FF9800; border:2px solid #EF6C00;"></div> Implementation</div>
<div style="display:flex; align-items:center; gap:6px;"><div style="width:14px; height:14px; border-radius:50%; background:#9C27B0; border:2px solid #7B1FA2;"></div> Artifact</div>
<div style="display:flex; align-items:center; gap:6px;"><div style="width:14px; height:14px; border-radius:50%; background:#F44336; border:2px solid #C62828;"></div> Heuristic</div>
<div style="display:flex; align-items:center; gap:6px;"><div style="width:14px; height:14px; border-radius:50%; background:#00BCD4; border:2px solid #0097A7;"></div> Environment</div>
<div style="display:flex; align-items:center; gap:6px;"><div style="width:14px; height:14px; border-radius:50%; background:#795548; border:2px solid #5D4037;"></div> Resource</div>
</div><noinclude>
== Usage ==
Add <code><nowiki>{{NetworkLegend}}</nowiki></code> before your network graph.
[[Category:Templates]]
</noinclude>
LEGENDTEMPLATE
    php maintenance/run.php edit.php \
        --user "$MW_ADMIN_USER" \
        --summary "Auto-created: Network legend template" \
        "Template:NetworkLegend" < /tmp/network_legend.txt
    rm /tmp/network_legend.txt
    echo "✓ Network Legend template created"
    
    # Create Common.js for network graph enhancements
    echo "📄 Creating Common.js for graph enhancements..."
    cat > /tmp/common_js.txt <<'COMMONJS'
/* Network Extension Enhancements - Filter external links, color by namespace */
(function() {
    'use strict';
    var nsColors = {
        'Workflow': {bg: '#4CAF50', border: '#2E7D32', font: '#1B5E20'},
        'Principle': {bg: '#2196F3', border: '#1565C0', font: '#0D47A1'},
        'Implementation': {bg: '#FF9800', border: '#EF6C00', font: '#E65100'},
        'Artifact': {bg: '#9C27B0', border: '#7B1FA2', font: '#4A148C'},
        'Heuristic': {bg: '#F44336', border: '#C62828', font: '#B71C1C'},
        'Environment': {bg: '#00BCD4', border: '#0097A7', font: '#006064'},
        'Resource': {bg: '#795548', border: '#5D4037', font: '#3E2723'}
    };
    var excludedPrefixes = ['Template:', 'Category:', 'MediaWiki:', 'Special:', 'File:'];
    if (window.vis && window.vis.DataSet) {
        var originalUpdate = window.vis.DataSet.prototype.update;
        window.vis.DataSet.prototype.update = function(data, senderId) {
            if (!Array.isArray(data)) data = [data];
            data = data.filter(function(item) {
                var id = item.id || '', to = item.to || '', from = item.from || '';
                if (id.indexOf('http://') === 0 || id.indexOf('https://') === 0) return false;
                if (to.indexOf('http://') === 0 || to.indexOf('https://') === 0) return false;
                if (from.indexOf('http://') === 0 || from.indexOf('https://') === 0) return false;
                for (var i = 0; i < excludedPrefixes.length; i++) {
                    if (id.indexOf(excludedPrefixes[i]) === 0) return false;
                    if (to.indexOf(excludedPrefixes[i]) === 0) return false;
                    if (from.indexOf(excludedPrefixes[i]) === 0) return false;
                }
                return true;
            }).map(function(item) {
                if (item.id && !item.from && !item.to) {
                    for (var ns in nsColors) {
                        if (item.id.indexOf(ns + ':') === 0) {
                            item.color = {background: nsColors[ns].bg, border: nsColors[ns].border,
                                highlight: {background: nsColors[ns].bg, border: nsColors[ns].border}};
                            item.font = item.font || {};
                            item.font.color = nsColors[ns].font;
                            break;
                        }
                    }
                }
                return item;
            });
            return originalUpdate.call(this, data, senderId);
        };
    }
})();
COMMONJS
    php maintenance/run.php edit.php \
        --user "$MW_ADMIN_USER" \
        --summary "Auto-created: Network graph enhancements" \
        "MediaWiki:Common.js" < /tmp/common_js.txt
    rm /tmp/common_js.txt
    echo "✓ Common.js created"
else
    echo "✓ Wiki already initialized (LocalSettings.php exists)"
fi

# Create API agent user if requested
# This user has sysop (admin) and bot privileges for automation
if [ "${MW_CREATE_AGENT:-false}" = "true" ] && [ -n "${MW_AGENT_USER:-}" ] && [ -n "${MW_AGENT_PASS:-}" ]; then
    echo "👤 Creating/updating API agent user..."
    php maintenance/run.php createAndPromote.php \
        --force "$MW_AGENT_USER" "$MW_AGENT_PASS" --sysop --bot 2>/dev/null || true
    echo "✓ API user '$MW_AGENT_USER' ready"
fi

# Import wiki pages from /wikis directory if mounted
# This loads .mediawiki files from data/wikis into MediaWiki
if [ -d "/wikis" ]; then
    /import_wikis.sh
fi

echo "🚀 Starting MediaWiki on ${MW_SITE_SERVER}"
exec docker-php-entrypoint "$@"

