<?php
// Script to create the API agent user
$services = MediaWiki\MediaWikiServices::getInstance();
$userFactory = $services->getUserFactory();
$userName = getenv('MW_AGENT_USER');
$userPass = getenv('MW_AGENT_PASS');

if (!$userName || !$userPass) {
    echo "ERROR: MW_AGENT_USER or MW_AGENT_PASS not set\n";
    exit(1);
}

// Canonicalize username (MediaWiki capitalizes first letter)
$userName = ucfirst($userName);
$user = $userFactory->newFromName($userName);

if (!$user) {
    echo "ERROR: Invalid username: $userName\n";
    exit(1);
}

if ($user->getId() === 0) {
    // User doesn't exist, create it
    $user->addToDatabase();
    $user->setPassword($userPass);
    $user->saveSettings();
    
    // Add to sysop and bot groups
    $groupManager = $services->getUserGroupManager();
    $groupManager->addUserToGroup($user, 'sysop');
    $groupManager->addUserToGroup($user, 'bot');
    
    echo "✓ Created API user: $userName (with sysop + bot rights)\n";
} else {
    // User exists, ensure it has proper rights
    $groupManager = $services->getUserGroupManager();
    $groups = $groupManager->getUserGroups($user);
    
    if (!in_array('sysop', $groups)) {
        $groupManager->addUserToGroup($user, 'sysop');
    }
    if (!in_array('bot', $groups)) {
        $groupManager->addUserToGroup($user, 'bot');
    }
    
    echo "✓ API user exists: $userName\n";
}
