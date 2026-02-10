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
$wgCargoAllowedSQLFunctions[] = '_pageName';
$wgCargoAllowedSQLFunctions[] = '_pageNamespace';
$wgCargoIgnoreBacklinks = true;  // Disable backlinks to avoid SQL bug with table aliasing
wfLoadExtension('VisualEditor');
wfLoadExtension('SyntaxHighlight_GeSHi');
wfLoadExtension('Math');
wfLoadExtension('Mermaid');
# wfLoadExtension('DynamicPageList3');  # Disabled - requires MW 1.44+
wfLoadExtension('Network');
wfLoadExtension('CategoryTree');

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

# Enable PAGESINNAMESPACE magic word for dynamic page counts
$wgAllowSlowParserFunctions = true;

# ============================================
# Semantic MediaWiki
# ============================================
enableSemantics();

# Disable SMW purge button (the ||| icon in page header)
$smwgPurgeEnabled = false;

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
            'hidden' => true,
            'physics' => false,
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
    
    // Only add to custom knowledge namespaces (excluding Workflow)
    $knowledgeNamespaces = [
        3000, // Principle
        // 3002, // Workflow - excluded, no page connections
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
    
    # Patch Network extension to show only outgoing links (not backlinks)
    # This removes 'linkshere' from the API query so only outgoing links are displayed
    echo "🔧 Patching Network extension for outgoing-only links..."
    sed -i "s/prop: \['links', 'linkshere', 'extlinks'\]/prop: ['links', 'extlinks']/" \
        /var/www/html/extensions/Network/resources/js/ApiPageConnectionRepo.js
    sed -i "/lhlimit: 'max',/d" \
        /var/www/html/extensions/Network/resources/js/ApiPageConnectionRepo.js
    echo "✓ Network extension patched"
    
    # Setup Semantic MediaWiki data store
    echo "🔄 Setting up Semantic MediaWiki..."
    php extensions/SemanticMediaWiki/maintenance/setupStore.php --skip-optimize --quiet
    echo "✓ Semantic MediaWiki ready"
    
    # Run database update to create tables for all extensions (Math, Cargo, etc.)
    echo "🔄 Running database update for extensions..."
    php maintenance/run.php update.php --quick 2>/dev/null || true
    echo "✓ Extension tables created"
    
    # Create Main Page with SMW queries
    echo "📄 Creating Main Page..."
    cat > /tmp/main_page.txt <<'MAINPAGE'
== Welcome to {{SITENAME}} ==

Your centralized playbook for '''Machine Learning''' and '''Data Engineering''' excellence. Discover expert-level implementation patterns, battle-tested best practices, and deep technical insights, all curated to accelerate your path from concept to production.

Want to connect this knowledge base to your AI agents? Follow the guide at [https://github.com/Leeroo-AI/kapso Kapso on GitHub].

== Browse by Category ==

{| class="wikitable" style="width:100%"
|-
! Category !! Description !! Browse
|-
| '''Workflows''' || Step-by-step processes and procedures || [[Special:AllPages/Workflow:|Browse All]]
|-
| '''Principles''' || Core ideas and foundational knowledge || [[Special:AllPages/Principle:|Browse All]]
|-
| '''Implementations''' || Code-level details and modules || [[Special:AllPages/Implementation:|Browse All]]
|-
| '''Heuristics''' || Best practices and guidelines || [[Special:AllPages/Heuristic:|Browse All]]
|-
| '''Environments''' || Setup and configuration guides || [[Special:AllPages/Environment:|Browse All]]
|}

== Recent Pages ==

=== Workflows ===
{{#cargo_query:tables=PageInfo|fields=_pageName|where=PageType='Workflow'|limit=10|format=ul}}

=== Principles ===
{{#cargo_query:tables=PageInfo|fields=_pageName|where=PageType='Principle'|limit=10|format=ul}}

=== Implementations ===
{{#cargo_query:tables=PageInfo|fields=_pageName|where=PageType='Implementation'|limit=10|format=ul}}

=== Heuristics ===
{{#cargo_query:tables=PageInfo|fields=_pageName|where=PageType='Heuristic'|limit=10|format=ul}}

=== Environments ===
{{#cargo_query:tables=PageInfo|fields=_pageName|where=PageType='Environment'|limit=10|format=ul}}
MAINPAGE
    
    php maintenance/run.php edit.php \
        --user "$MW_ADMIN_USER" \
        --summary "Auto-created: Main page with DPL index" \
        "Main Page" < /tmp/main_page.txt
    rm /tmp/main_page.txt
    echo "✓ Main Page created"

    # Create Category pages for CategoryTree to work
    echo "📄 Creating category pages..."
    echo "Pages related to step-by-step processes and procedures." | php maintenance/run.php edit.php --user "$MW_ADMIN_USER" --summary "Auto-created category" "Category:Workflows"
    echo "Pages related to core ideas and foundational knowledge." | php maintenance/run.php edit.php --user "$MW_ADMIN_USER" --summary "Auto-created category" "Category:Principles"
    echo "Pages related to code-level details and modules." | php maintenance/run.php edit.php --user "$MW_ADMIN_USER" --summary "Auto-created category" "Category:Implementations"
    echo "Pages related to best practices and guidelines." | php maintenance/run.php edit.php --user "$MW_ADMIN_USER" --summary "Auto-created category" "Category:Heuristics"
    echo "Pages related to setup and configuration guides." | php maintenance/run.php edit.php --user "$MW_ADMIN_USER" --summary "Auto-created category" "Category:Environments"
    echo "✓ Category pages created"

    # Create PageInfo template for Cargo table
    echo "📄 Creating PageInfo Cargo template..."
    cat > /tmp/pageinfo_template.txt <<'PAGEINFO'
<noinclude>
{{#cargo_declare:_table=PageInfo
|PageType=String
|PageTitle=String
}}
This template stores page metadata for querying.
[[Category:Templates]]
</noinclude><includeonly>{{#cargo_store:_table=PageInfo
|PageType={{{type|}}}
|PageTitle={{{title|}}}
}}</includeonly>
PAGEINFO
    php maintenance/run.php edit.php \
        --user "$MW_ADMIN_USER" \
        --summary "Auto-created: PageInfo Cargo template" \
        "Template:PageInfo" < /tmp/pageinfo_template.txt
    rm /tmp/pageinfo_template.txt
    echo "✓ PageInfo template created"

    # Create Cargo table with proper schema (including internal columns like _pageID)
    # Note: Don't use --replacement flag as it requires an existing table to swap with
    echo "🔄 Initializing Cargo PageInfo table..."
    php /var/www/html/extensions/Cargo/maintenance/cargoRecreateData.php --table PageInfo 2>&1 || true
    echo "✓ Cargo PageInfo table initialized"

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

    # Create Common.css for UI customizations
    echo "📄 Creating Common.css..."
    cat > /tmp/common_css.txt <<'COMMONCSS'
/* Hide Semantic MediaWiki purge button */
.smw-purge {
    display: none !important;
}

/* Hide SMW entity examiner / vertical bar loader (the ||| icon) */
.smw-indicator-vertical-bar-loader,
.smw-entity-examiner {
    display: none !important;
}

/* Hide Vector 2022 page tools dropdown */
#vector-page-tools-dropdown {
    display: none !important;
}
COMMONCSS
    php maintenance/run.php edit.php \
        --user "$MW_ADMIN_USER" \
        --summary "Auto-created: Hide SMW purge button" \
        "MediaWiki:Common.css" < /tmp/common_css.txt
    rm /tmp/common_css.txt
    echo "✓ Common.css created"
else
    echo "✓ Wiki already initialized (LocalSettings.php exists)"
fi

# Apply wiki permissions (private wiki with tiered access)
echo "🔒 Applying wiki permissions..."
if ! grep -q "WIKI PERMISSIONS" /var/www/html/LocalSettings.php; then
    cat >> /var/www/html/LocalSettings.php <<'PERMISSIONS'

# ============================================
# WIKI PERMISSIONS
# ============================================
# Private wiki: login required to view
$wgGroupPermissions['*']['read'] = false;
$wgGroupPermissions['*']['edit'] = false;
$wgGroupPermissions['*']['createaccount'] = true;

# Allow anonymous users to access login and account creation pages
$wgWhitelistRead = [
    'Special:UserLogin',
    'Special:CreateAccount',
    'Special:PasswordReset',
];

# Users: can read and edit (talk pages only due to namespace protection below)
$wgGroupPermissions['user']['read'] = true;
$wgGroupPermissions['user']['edit'] = true;
$wgGroupPermissions['user']['createpage'] = false;
$wgGroupPermissions['user']['createtalk'] = true;

# Admins: full edit access (bypasses namespace protection)
$wgGroupPermissions['sysop']['edit'] = true;
$wgGroupPermissions['sysop']['createpage'] = true;
$wgGroupPermissions['sysop']['editprotected'] = true;

# Protect main content namespaces - only users with 'editprotected' can edit
# NS_MAIN (0), NS_PROJECT (4), NS_TEMPLATE (10), NS_CATEGORY (14)
# Plus custom namespaces (3000-3012)
$wgNamespaceProtection[NS_MAIN] = ['editprotected'];
$wgNamespaceProtection[NS_PROJECT] = ['editprotected'];
$wgNamespaceProtection[NS_TEMPLATE] = ['editprotected'];
$wgNamespaceProtection[NS_CATEGORY] = ['editprotected'];
$wgNamespaceProtection[NS_PRINCIPLE] = ['editprotected'];
$wgNamespaceProtection[NS_WORKFLOW] = ['editprotected'];
$wgNamespaceProtection[NS_IMPLEMENTATION] = ['editprotected'];
$wgNamespaceProtection[NS_ARTIFACT] = ['editprotected'];
$wgNamespaceProtection[NS_HEURISTIC] = ['editprotected'];
$wgNamespaceProtection[NS_ENVIRONMENT] = ['editprotected'];
$wgNamespaceProtection[NS_RESOURCE] = ['editprotected'];

# ============================================
# SIGNUP REQUIREMENTS
# ============================================
# Make email and real name required, add company name field

# Add Company Name field to signup form
$wgHooks['AuthChangeFormFields'][] = function ($requests, $fieldInfo, &$formDescriptor, $action) {
    if ($action !== 'create') {
        return true;
    }

    // Make email required and update label
    if (isset($formDescriptor['email'])) {
        $formDescriptor['email']['required'] = true;
        $formDescriptor['email']['label'] = 'Email address';
        unset($formDescriptor['email']['label-message']);
    }

    // Make real name required and update label
    if (isset($formDescriptor['realname'])) {
        $formDescriptor['realname']['required'] = true;
        $formDescriptor['realname']['label'] = 'Real name';
        $formDescriptor['realname']['placeholder'] = 'Enter your real name';
        unset($formDescriptor['realname']['label-message']);
    }

    // Add Company Name field after real name
    $formDescriptor['companyname'] = [
        'type' => 'text',
        'label' => 'Company name',
        'help-message' => 'prefs-help-company-name',
        'required' => true,
        'weight' => 35,  // Position after realname (30) but before email (40)
    ];

    return true;
};

# Validate required fields and save company name on account creation
$wgHooks['LocalUserCreated'][] = function ($user, $autocreated) {
    if ($autocreated) {
        return true;
    }

    global $wgRequest;
    $companyName = $wgRequest->getText('wpcompanyname');

    $userOptionsManager = MediaWiki\MediaWikiServices::getInstance()->getUserOptionsManager();

    // Store company name in user preferences
    if (!empty($companyName)) {
        $userOptionsManager->setOption($user, 'companyname', $companyName);
    }

    // Generate Leeroopedia API key: lp_<user_id>_<32_hex_chars>
    $apiKey = 'lp_' . $user->getId() . '_' . bin2hex(random_bytes(16));
    $userOptionsManager->setOption($user, 'leeroopedia_api_key', $apiKey);

    $userOptionsManager->saveOptions($user);

    return true;
};

# Add company name to user preferences for later editing
$wgHooks['GetPreferences'][] = function ($user, &$preferences) {
    $preferences['companyname'] = [
        'type' => 'text',
        'label' => 'Company name',
        'section' => 'personal/info',
        'help' => 'Your company or organization name.',
    ];

    // Display Leeroopedia API key with show/copy buttons
    $userOptionsManager = MediaWiki\MediaWikiServices::getInstance()->getUserOptionsManager();
    $apiKey = $userOptionsManager->getOption($user, 'leeroopedia_api_key', '');
    $maskedKey = $apiKey ? str_repeat('•', strlen($apiKey)) : 'No API key generated';

    $apiKeyHtml = '<div style="display: flex; align-items: center; gap: 10px;">
        <code id="apikey-display" style="font-family: monospace; padding: 4px 8px; background: #f5f5f5; border-radius: 4px;">' . htmlspecialchars($maskedKey) . '</code>
        <input type="hidden" id="apikey-value" value="' . htmlspecialchars($apiKey) . '">
        <button type="button" id="apikey-toggle" onclick="toggleApiKey()" style="padding: 4px 12px; cursor: pointer; border: 1px solid #a2a9b1; border-radius: 4px; background: #f8f9fa;">Show</button>
        <button type="button" onclick="copyApiKey()" style="padding: 4px 12px; cursor: pointer; border: 1px solid #a2a9b1; border-radius: 4px; background: #f8f9fa;">Copy</button>
    </div>
    <script>
    var apiKeyVisible = false;
    function toggleApiKey() {
        var display = document.getElementById("apikey-display");
        var value = document.getElementById("apikey-value").value;
        var btn = document.getElementById("apikey-toggle");
        if (apiKeyVisible) {
            display.textContent = "' . str_repeat('•', strlen($apiKey)) . '";
            btn.textContent = "Show";
            apiKeyVisible = false;
        } else {
            display.textContent = value;
            btn.textContent = "Hide";
            apiKeyVisible = true;
        }
    }
    function copyApiKey() {
        var value = document.getElementById("apikey-value").value;
        navigator.clipboard.writeText(value).then(function() {
            var btn = event.target;
            var orig = btn.textContent;
            btn.textContent = "Copied!";
            setTimeout(function() { btn.textContent = orig; }, 1500);
        });
    }
    </script>';

    $preferences['leeroopedia_api_key'] = [
        'type' => 'info',
        'label' => 'Leeroopedia API Key',
        'default' => $apiKeyHtml,
        'section' => 'personal/info',
        'raw' => true,
        'help' => 'Use this key to access the Leeroopedia Content API.',
    ];

    return true;
};

# Define messages for company name
$wgHooks['MessageCacheReplace'][] = function (&$cache) {
    return true;
};

# Custom messages
$wgExtensionMessagesFiles['LeeroopediaCustom'] = null;
$wgMessagesDirs['LeeroopediaCustom'] = [];
$wgHooks['LoadExtensionSchemaUpdates'][] = function () {
    return true;
};

# Inline message definitions
$wgHooks['MessagesPreLoad'][] = function ($title, &$message, $code) {
    $customMessages = [
        'prefs-companyname' => 'Company name',
        'prefs-help-company-name' => 'Your company or organization name.',
        'createacct-realname' => 'Real name',
        'prefs-help-realname' => 'Your real name for attribution.',
    ];
    if (isset($customMessages[$title])) {
        $message = $customMessages[$title];
    }
    return true;
};
PERMISSIONS
    echo "✓ Wiki permissions applied"
else
    echo "✓ Wiki permissions already set"
fi

# Configure email via Resend HTTP API (for proper HTML support)
if [ -n "${SMTP_PASS:-}" ]; then
    echo "📧 Configuring Resend email API..."
    if ! grep -q "RESEND EMAIL CONFIGURATION" /var/www/html/LocalSettings.php; then
        cat >> /var/www/html/LocalSettings.php <<EMAILCONFIG
# ============================================
# RESEND EMAIL CONFIGURATION
# ============================================
\$wgEnableEmail = true;
\$wgEnableUserEmail = true;
\$wgAllowHTMLEmail = true;
\$wgUserEmailUseReplyTo = true;
\$wgPasswordSender = '${SMTP_FROM:-wiki@localhost}';
\$wgEmergencyContact = '${SMTP_FROM:-wiki@localhost}';

# Custom mailer using Resend HTTP API with HTML email templates
\$wgHooks['AlternateUserMailer'][] = function (\$headers, \$to, \$from, \$subject, \$body) {
    \$apiKey = '${SMTP_PASS}';
    \$siteServer = '${MW_SITE_SERVER}';

    // Parse recipients
    \$toAddresses = [];
    if (is_array(\$to)) {
        foreach (\$to as \$recipient) {
            if (is_object(\$recipient) && method_exists(\$recipient, 'toString')) {
                \$toAddresses[] = \$recipient->toString();
            } elseif (is_string(\$recipient)) {
                \$toAddresses[] = \$recipient;
            }
        }
    } else {
        \$toAddresses[] = is_object(\$to) ? \$to->toString() : (string)\$to;
    }

    // Parse from address
    \$fromAddress = is_object(\$from) ? \$from->toString() : (string)\$from;
    if (empty(\$fromAddress)) {
        \$fromAddress = '${SMTP_FROM:-wiki@localhost}';
    }

    // Transform email body to HTML based on email type
    \$htmlBody = null;
    \$newSubject = \$subject;

    // Email confirmation
    if (strpos(\$body, 'Special:ConfirmEmail') !== false) {
        \$newSubject = 'Welcome to Leeroopedia';
        // Extract confirmation URL
        preg_match('/(http[^\s]+Special:ConfirmEmail[^\s]+)/', \$body, \$matches);
        \$confirmUrl = \$matches[1] ?? '#';
        // Extract expiry
        preg_match('/expire[s]? at ([^\\.]+)/', \$body, \$expiryMatch);
        \$expiry = \$expiryMatch[1] ?? '7 days';

        \$htmlBody = "<!DOCTYPE html>
<html><head><meta charset='UTF-8'/><style>
body{background:#fff;margin:0;padding:0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;color:#000}
.container{max-width:600px;margin:0 auto;padding:20px}
.header{text-align:center;padding:20px 0}
.logo{height:50px;margin-bottom:10px}
.content{background:#fff;padding:20px;border-radius:6px;border:1px solid #eee}
h1{font-size:24px;margin-bottom:16px;color:#000;text-align:center}
p{line-height:1.6;font-size:16px;margin:10px 0}
.button-wrapper{text-align:center;margin:20px 0}
.button{display:inline-block;background:#000;color:#fff;text-decoration:none;padding:12px 24px;border-radius:4px;font-weight:bold}
.info-box{margin:20px 0;padding:15px;background:#f5f5f5;border-radius:6px}
.footer{text-align:center;font-size:14px;color:#999;margin-top:30px}
a{color:#000;text-decoration:none}
</style></head><body>
<div class='container'><div class='content'>
<div class='header'>
<img src='https://xwipmlfvxtqnizhwoppi.supabase.co/storage/v1/object/public/logo//400dpiLogoCropped.png' alt='Leeroopedia' class='logo'/>
<h1>Welcome to Leeroopedia!</h1>
</div>
<p>Hi there,</p>
<p>Thank you for joining us! We are thrilled to welcome you to Leeroopedia.</p>
<p>Leeroopedia is your centralized playbook for <strong>Machine Learning</strong> and <strong>Data Engineering</strong>. We have built this platform to serve as a comprehensive knowledge wiki, gathering expert-level implementation patterns, best practices, and deep industry insights all in one place.</p>
<p>To get started, please confirm your email address:</p>
<div class='button-wrapper'><a href='" . htmlspecialchars(\$confirmUrl) . "' class='button'>Confirm Your Email</a></div>
<p style='text-align:center;color:#999;font-size:14px;margin-top:15px'>This link will expire at " . htmlspecialchars(\$expiry) . "</p>
<div class='info-box'>
<p style='margin:0'><strong>Connect with Autonomous Agents</strong></p>
<p style='margin:10px 0 0 0'>To connect Leeroopedia to your own agents, check out the <a href='https://github.com/Leeroo-AI/kapso'><strong>Kapso guide on GitHub</strong></a>.</p>
</div>
</div>
<div class='footer'><p>Welcome aboard!</p><p>2026 Leeroo. All rights reserved.</p></div>
</div></body></html>";
    }

    // Password reset
    if (strpos(\$body, 'Temporary password') !== false || strpos(\$body, 'reset of your') !== false) {
        \$newSubject = 'Reset your Leeroopedia password';
        // Extract username and temp password
        preg_match('/Username:\\s*([^\\n]+)/s', \$body, \$userMatch);
        preg_match('/Temporary password:\\s*([^\\n]+)/s', \$body, \$passMatch);
        \$username = trim(\$userMatch[1] ?? '');
        \$tempPass = trim(\$passMatch[1] ?? '');

        \$htmlBody = "<!DOCTYPE html>
<html><head><meta charset='UTF-8'/><style>
body{background:#fff;margin:0;padding:0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;color:#000}
.container{max-width:600px;margin:0 auto;padding:20px}
.header{text-align:center;padding:20px 0}
.logo{height:50px;margin-bottom:10px}
.content{background:#fff;padding:20px;border-radius:6px;border:1px solid #eee}
h1{font-size:24px;margin-bottom:16px;color:#000;text-align:center}
p{line-height:1.6;font-size:16px;margin:10px 0}
.password-box{margin:20px 0;padding:20px;background:#fafafa;border:2px dashed #000;text-align:center;border-radius:6px}
.password-code{display:block;font-size:24px;font-weight:bold;letter-spacing:2px;color:#000;margin:10px 0;font-family:monospace}
.button-wrapper{text-align:center;margin:20px 0}
.button{display:inline-block;background:#000;color:#fff;text-decoration:none;padding:12px 24px;border-radius:4px;font-weight:bold}
.warning-box{margin:20px 0;padding:15px;background:#fff3cd;border:1px solid #ffc107;border-radius:6px}
.footer{text-align:center;font-size:14px;color:#999;margin-top:30px}
a{color:#000;text-decoration:none}
</style></head><body>
<div class='container'><div class='content'>
<div class='header'>
<img src='https://xwipmlfvxtqnizhwoppi.supabase.co/storage/v1/object/public/logo//400dpiLogoCropped.png' alt='Leeroopedia' class='logo'/>
<h1>Password Reset</h1>
</div>
<p>Hello,</p>
<p>We received a request to reset your password for Leeroopedia.</p>
<div class='password-box'>
<p><strong>Your Temporary Password</strong></p>
<span class='password-code'>" . htmlspecialchars(\$tempPass) . "</span>
<p style='font-size:14px;color:#666'>Username: " . htmlspecialchars(\$username) . "</p>
</div>
<div class='button-wrapper'><a href='" . \$siteServer . "/index.php/Special:UserLogin' class='button'>Log In Now</a></div>
<div class='warning-box'><p style='margin:0'><strong>Note:</strong> This temporary password will expire in 7 days.</p></div>
<p style='color:#666;font-size:14px'>If you did not request a password reset, you can safely ignore this email. Your account remains secure.</p>
</div>
<div class='footer'><p>Best regards,<br>The Leeroo Team</p><p>2026 Leeroo. All rights reserved.</p></div>
</div></body></html>";
    }

    // Build payload
    \$payload = [
        'from' => \$fromAddress,
        'to' => \$toAddresses,
        'subject' => \$newSubject,
    ];

    if (\$htmlBody !== null) {
        \$payload['html'] = \$htmlBody;
    } else {
        \$payload['text'] = \$body;
    }

    // Send via Resend API
    \$ch = curl_init('https://api.resend.com/emails');
    curl_setopt(\$ch, CURLOPT_RETURNTRANSFER, true);
    curl_setopt(\$ch, CURLOPT_POST, true);
    curl_setopt(\$ch, CURLOPT_POSTFIELDS, json_encode(\$payload));
    curl_setopt(\$ch, CURLOPT_HTTPHEADER, [
        'Authorization: Bearer ' . \$apiKey,
        'Content-Type: application/json',
    ]);

    \$response = curl_exec(\$ch);
    \$httpCode = curl_getinfo(\$ch, CURLINFO_HTTP_CODE);
    curl_close(\$ch);

    if (\$httpCode >= 200 && \$httpCode < 300) {
        return true;
    }

    wfDebugLog('email', "Resend API error (\$httpCode): \$response");
    return false;
};
EMAILCONFIG
        echo "✓ Resend email API configured"
    else
        echo "✓ Email already configured"
    fi
else
    echo "ℹ️  Email not configured (set SMTP_PASS with Resend API key)"
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
# NOTE: The actual import is triggered by start.sh after Apache is up.
# We skip running it here to avoid duplicate imports on large wikis.
if [ -d "/wikis" ]; then
    echo "ℹ️  Wiki pages will be imported after Apache starts (via start.sh)"
fi

# Start background job to refresh site stats periodically (for accurate page counts)
echo "🔄 Starting background stats refresh job..."
(
    sleep 60  # Wait for Apache to start and initial import to complete
    while true; do
        php /var/www/html/maintenance/run.php initSiteStats.php --update > /dev/null 2>&1
        sleep 30
    done
) &

echo "🚀 Starting MediaWiki on ${MW_SITE_SERVER}"
exec docker-php-entrypoint "$@"

