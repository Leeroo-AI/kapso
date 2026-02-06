<?php
/**
 * Endpoint to trigger immediate site stats and Cargo table refresh.
 * Called by sync service after page deletions to update:
 * - PAGESINNAMESPACE counts (site_stats table)
 * - Cargo PageInfo table (remove orphaned entries)
 * - Main_Page parser cache
 */

// Basic auth check using agent credentials
$expectedUser = getenv('MW_AGENT_USER') ?: 'agent';
$expectedPass = getenv('MW_AGENT_PASS') ?: getenv('WIKI_AGENT_PASSWORD');

// Verify request is authorized via token query param
$providedToken = $_GET['token'] ?? '';
$expectedToken = md5($expectedUser . $expectedPass);

if (empty($providedToken) || $providedToken !== $expectedToken) {
    http_response_code(401);
    header('Content-Type: application/json');
    die(json_encode(['error' => 'Unauthorized']));
}

$results = [];

// 1. Run initSiteStats.php to refresh site_stats table
$output = [];
$returnCode = 0;
exec('php /var/www/html/maintenance/run.php initSiteStats.php --update 2>&1', $output, $returnCode);
$results['site_stats'] = [
    'success' => $returnCode === 0,
    'output' => implode("\n", $output)
];

// Database operations (Cargo cleanup + page touch)
try {
    // Get database credentials from environment
    $dbHost = getenv('MW_DB_HOST') ?: 'db';
    $dbUser = getenv('MW_DB_USER') ?: 'mediawiki';
    $dbPass = getenv('MW_DB_PASSWORD') ?: '';
    $dbName = getenv('MW_DB_NAME') ?: 'wiki';

    $db = new mysqli($dbHost, $dbUser, $dbPass, $dbName);

    if ($db->connect_error) {
        throw new Exception("DB connection failed: " . $db->connect_error);
    }

    // 2. Delete Cargo PageInfo entries where the page no longer exists
    $sql = "DELETE c FROM cargo__PageInfo c
            LEFT JOIN page p ON c._pageID = p.page_id
            WHERE p.page_id IS NULL";

    if ($db->query($sql)) {
        $results['cargo_cleanup'] = [
            'success' => true,
            'deleted_rows' => $db->affected_rows
        ];
    } else {
        $results['cargo_cleanup'] = [
            'success' => false,
            'error' => $db->error
        ];
    }

    // 3. Invalidate Main_Page parser cache by touching it in the database
    // This forces MediaWiki to re-parse the page including expensive functions like PAGESINNAMESPACE
    $touchSql = "UPDATE page SET page_touched = DATE_FORMAT(NOW(), '%Y%m%d%H%i%s') WHERE page_title = 'Main_Page' AND page_namespace = 0";
    if ($db->query($touchSql)) {
        $results['main_page_touch'] = [
            'success' => true,
            'affected_rows' => $db->affected_rows
        ];
    } else {
        $results['main_page_touch'] = [
            'success' => false,
            'error' => $db->error
        ];
    }

    $db->close();
} catch (Exception $e) {
    $results['db_error'] = $e->getMessage();
}

// 4. Refresh Cargo PageInfo table to pick up new pages
$cargoOutput = [];
$cargoReturnCode = 0;
// Drop any leftover replacement table first
$db = new mysqli($dbHost, $dbUser, $dbPass, $dbName);
if (!$db->connect_error) {
    $db->query("DROP TABLE IF EXISTS cargo__PageInfo__NEXT");
    $db->close();
}
exec('yes | php /var/www/html/extensions/Cargo/maintenance/cargoRecreateData.php --table PageInfo 2>&1', $cargoOutput, $cargoReturnCode);
$results['cargo_refresh'] = [
    'success' => $cargoReturnCode === 0,
    'output' => implode("\n", $cargoOutput)
];

// 5. Also purge via purgeList for CDN/proxy cache
$purgeOutput = [];
$purgeReturnCode = 0;
exec('echo "Main_Page" | php /var/www/html/maintenance/run.php purgeList.php 2>&1', $purgeOutput, $purgeReturnCode);
$results['main_page_purge'] = [
    'success' => $purgeReturnCode === 0,
    'output' => implode("\n", $purgeOutput)
];

// Return result
header('Content-Type: application/json');
echo json_encode([
    'success' => ($results['site_stats']['success'] ?? false)
             && ($results['cargo_cleanup']['success'] ?? false)
             && ($results['main_page_touch']['success'] ?? false),
    'results' => $results,
    'timestamp' => date('c')
]);
