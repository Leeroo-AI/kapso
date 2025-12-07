#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wiki_editor.py - MediaWiki editing client

A clean, modular client for editing MediaWiki instances with support for:
- Multiple wiki configurations
- Common editing operations (create, edit, append, prepend, delete, move)
- Batch operations
- Configuration file support
- Command-line interface

Usage:
    # Using command-line arguments
    wiki_editor.py --url https://wiki.example.com --user Bot --password pass123 --action edit --title "Test Page" --text "Content"
    
    # Using config file
    wiki_editor.py --config config/config.json --wiki production --action edit --title "Test Page" --text "Content"
    
    # Batch operations from file
    wiki_editor.py --config config/config.json --wiki test --batch operations.json
"""

import argparse
import json
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import requests
from urllib.parse import urlparse, urlunparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class WikiEditor:
    """
    MediaWiki editing client with clean interface for common operations
    """
    
    def __init__(self, 
                 url: str, 
                 username: str, 
                 password: str,
                 verify_ssl: bool = True,
                 timeout: int = 30,
                 subdomain: Optional[str] = None):
        """
        Initialize the wiki editor
        
        Args:
            url: Wiki base URL (e.g., 'https://wiki.example.com')
            username: MediaWiki username or bot username
            password: MediaWiki password or bot password
            verify_ssl: Whether to verify SSL certificates
            timeout: Request timeout in seconds
            subdomain: Optional subdomain for routing (e.g., 'ml', 'de')
        """
        self.base_url = url.rstrip("/")
        self.username = username
        self.password = password
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "WikiEditor/1.0 (Python MediaWiki Client)"
        self.session.verify = verify_ssl
        self.csrf_token = None
        self._api_url = None
        
        # Apply subdomain routing if specified
        if subdomain:
            self._apply_subdomain(subdomain)
        
        # Track operation statistics
        self.stats = {
            'operations': 0,
            'successful': 0,
            'failed': 0,
            'errors': []
        }
    
    def _apply_subdomain(self, subdomain: str) -> None:
        """
        Apply subdomain routing for multi-wiki setups
        
        Args:
            subdomain: Subdomain to route to (e.g., 'ml', 'de', 'quant')
        """
        try:
            parsed = urlparse(self.base_url)
            host_port = parsed.netloc
            
            if not host_port:
                return
            
            # Handle port if present
            if ":" in host_port and host_port.count(":") == 1:
                host, port = host_port.split(":", 1)
                port_suffix = f":{port}"
            else:
                host = host_port
                port_suffix = ""
            
            # Apply subdomain logic
            labels = host.split(".")
            if len(labels) >= 3:
                # Replace first label with subdomain
                new_host = ".".join([subdomain] + labels[1:])
            elif len(labels) == 2:
                # Prepend subdomain
                new_host = ".".join([subdomain] + labels)
            else:
                # Single label or IP - keep as is
                return
            
            # Update base URL
            final_host = f"{new_host}{port_suffix}"
            self.base_url = urlunparse((
                parsed.scheme or "https",
                final_host,
                parsed.path.rstrip("/"),
                "", "", ""
            )).rstrip("/")
            
            # Set Host header for routing
            self.session.headers["Host"] = new_host
            logger.info(f"Applied subdomain routing: {self.base_url}")
            
        except Exception as e:
            logger.warning(f"Failed to apply subdomain: {e}")
    
    def _resolve_api_endpoint(self) -> str:
        """
        Resolve the API endpoint by trying common paths
        
        Returns:
            API endpoint URL
        
        Raises:
            RuntimeError: If API endpoint cannot be found
        """
        # Try common MediaWiki API paths
        api_paths = ["/api.php", "/w/api.php", "/mediawiki/api.php"]
        
        for path in api_paths:
            url = f"{self.base_url}{path}"
            try:
                # Try to get login token as test
                response = self.session.get(
                    url,
                    params={
                        "action": "query",
                        "meta": "tokens",
                        "type": "login",
                        "format": "json"
                    },
                    timeout=self.timeout,
                    allow_redirects=True
                )
                response.raise_for_status()
                data = response.json()
                
                # Check if we got a valid response with token
                if data.get("query", {}).get("tokens", {}).get("logintoken"):
                    self._api_url = url
                    logger.info(f"Resolved API endpoint: {url}")
                    return url
                    
            except Exception as e:
                logger.debug(f"API endpoint {url} failed: {e}")
                continue
        
        raise RuntimeError(
            f"Could not resolve API endpoint for {self.base_url}. "
            f"Tried paths: {', '.join(api_paths)}"
        )
    
    @property
    def api_url(self) -> str:
        """Get the API URL, resolving if necessary"""
        if not self._api_url:
            self._resolve_api_endpoint()
        return self._api_url
    
    def login(self) -> bool:
        """
        Login to MediaWiki
        
        Returns:
            True if login successful, False otherwise
        """
        try:
            # Get login token
            response = self.session.get(
                self.api_url,
                params={
                    "action": "query",
                    "meta": "tokens",
                    "type": "login",
                    "format": "json"
                },
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            login_token = data["query"]["tokens"]["logintoken"]
            
            # Perform login
            response = self.session.post(
                self.api_url,
                data={
                    "action": "login",
                    "format": "json",
                    "lgname": self.username,
                    "lgpassword": self.password,
                    "lgtoken": login_token
                },
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            
            if data.get("login", {}).get("result") != "Success":
                logger.error(f"Login failed: {data}")
                return False
            
            # Get CSRF token for editing
            response = self.session.get(
                self.api_url,
                params={
                    "action": "query",
                    "meta": "tokens",
                    "type": "csrf",
                    "format": "json"
                },
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            self.csrf_token = data["query"]["tokens"]["csrftoken"]
            
            logger.info(f"Successfully logged in as {self.username}")
            return True
            
        except Exception as e:
            logger.error(f"Login failed: {e}")
            return False
    
    def edit_page(self, 
                  title: str, 
                  text: Optional[str] = None,
                  summary: str = "",
                  append: Optional[str] = None,
                  prepend: Optional[str] = None,
                  createonly: bool = False,
                  bot: bool = True,
                  minor: bool = False) -> Dict[str, Any]:
        """
        Edit or create a wiki page
        
        Args:
            title: Page title
            text: Full page text (replaces existing content)
            summary: Edit summary
            append: Text to append to page
            prepend: Text to prepend to page
            createonly: Only create if page doesn't exist
            bot: Mark edit as bot edit
            minor: Mark edit as minor
            
        Returns:
            API response dictionary
        """
        if not self.csrf_token:
            if not self.login():
                return {"error": "Login failed"}
        
        # Build request payload
        payload = {
            "action": "edit",
            "format": "json",
            "assert": "user",
            "title": title,
            "summary": summary or f"Edited via WikiEditor at {datetime.now().isoformat()}",
            "token": self.csrf_token
        }
        
        # Handle content modification type
        if append is not None:
            payload["appendtext"] = append
            logger.info(f"Appending to page: {title}")
        elif prepend is not None:
            payload["prependtext"] = prepend
            logger.info(f"Prepending to page: {title}")
        else:
            payload["text"] = text or ""
            logger.info(f"Editing page: {title}")
        
        # Set flags
        if createonly:
            payload["createonly"] = 1
        if bot:
            payload["bot"] = 1
        if minor:
            payload["minor"] = 1
        
        try:
            response = self.session.post(
                self.api_url,
                data=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            
            if "error" in data:
                logger.error(f"Edit failed for {title}: {data['error']}")
                self.stats['failed'] += 1
                self.stats['errors'].append(data['error'])
            else:
                logger.info(f"Successfully edited {title}")
                self.stats['successful'] += 1
            
            self.stats['operations'] += 1
            return data
            
        except Exception as e:
            logger.error(f"Edit request failed for {title}: {e}")
            self.stats['failed'] += 1
            self.stats['operations'] += 1
            return {"error": str(e)}
    
    def move_page(self, 
                  from_title: str, 
                  to_title: str,
                  reason: str = "",
                  noredirect: bool = False,
                  movetalk: bool = True) -> Dict[str, Any]:
        """
        Move/rename a wiki page
        
        Args:
            from_title: Current page title
            to_title: New page title
            reason: Move reason
            noredirect: Don't create redirect (requires permission)
            movetalk: Also move talk page
            
        Returns:
            API response dictionary
        """
        if not self.csrf_token:
            if not self.login():
                return {"error": "Login failed"}
        
        payload = {
            "action": "move",
            "format": "json",
            "from": from_title,
            "to": to_title,
            "reason": reason or f"Moved via WikiEditor at {datetime.now().isoformat()}",
            "token": self.csrf_token
        }
        
        if noredirect:
            payload["noredirect"] = 1
        if movetalk:
            payload["movetalk"] = 1
        
        try:
            response = self.session.post(
                self.api_url,
                data=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            
            if "error" in data:
                logger.error(f"Move failed {from_title} -> {to_title}: {data['error']}")
                self.stats['failed'] += 1
                self.stats['errors'].append(data['error'])
            else:
                logger.info(f"Successfully moved {from_title} -> {to_title}")
                self.stats['successful'] += 1
            
            self.stats['operations'] += 1
            return data
            
        except Exception as e:
            logger.error(f"Move request failed: {e}")
            self.stats['failed'] += 1
            self.stats['operations'] += 1
            return {"error": str(e)}
    
    def delete_page(self, title: str, reason: str = "") -> Dict[str, Any]:
        """
        Delete a wiki page (requires delete permission)
        
        Args:
            title: Page title to delete
            reason: Deletion reason
            
        Returns:
            API response dictionary
        """
        if not self.csrf_token:
            if not self.login():
                return {"error": "Login failed"}
        
        payload = {
            "action": "delete",
            "format": "json",
            "title": title,
            "reason": reason or f"Deleted via WikiEditor at {datetime.now().isoformat()}",
            "token": self.csrf_token
        }
        
        try:
            response = self.session.post(
                self.api_url,
                data=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            
            if "error" in data:
                logger.error(f"Delete failed for {title}: {data['error']}")
                self.stats['failed'] += 1
                self.stats['errors'].append(data['error'])
            else:
                logger.info(f"Successfully deleted {title}")
                self.stats['successful'] += 1
            
            self.stats['operations'] += 1
            return data
            
        except Exception as e:
            logger.error(f"Delete request failed for {title}: {e}")
            self.stats['failed'] += 1
            self.stats['operations'] += 1
            return {"error": str(e)}
    
    def get_page_content(self, title: str, format: str = "wikitext") -> Optional[str]:
        """
        Get page content
        
        Args:
            title: Page title
            format: Content format ('wikitext' or 'html')
            
        Returns:
            Page content or None if error
        """
        try:
            prop = "wikitext" if format == "wikitext" else "text"
            response = self.session.get(
                self.api_url,
                params={
                    "action": "parse",
                    "page": title,
                    "prop": prop,
                    "format": "json"
                },
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            
            if "error" in data:
                logger.error(f"Failed to get content for {title}: {data['error']}")
                return None
            
            content_key = "wikitext" if format == "wikitext" else "text"
            return data["parse"][content_key]["*"]
            
        except Exception as e:
            logger.error(f"Failed to fetch page {title}: {e}")
            return None
    
    def batch_operations(self, operations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute multiple operations in batch
        
        Args:
            operations: List of operation dictionaries with 'action' and parameters
            
        Returns:
            List of results for each operation
        """
        results = []
        
        for op in operations:
            action = op.get("action")
            
            if action == "edit":
                result = self.edit_page(
                    title=op.get("title"),
                    text=op.get("text"),
                    summary=op.get("summary", ""),
                    append=op.get("append"),
                    prepend=op.get("prepend"),
                    createonly=op.get("createonly", False),
                    bot=op.get("bot", True),
                    minor=op.get("minor", False)
                )
            elif action == "move":
                result = self.move_page(
                    from_title=op.get("from"),
                    to_title=op.get("to"),
                    reason=op.get("reason", ""),
                    noredirect=op.get("noredirect", False),
                    movetalk=op.get("movetalk", True)
                )
            elif action == "delete":
                result = self.delete_page(
                    title=op.get("title"),
                    reason=op.get("reason", "")
                )
            else:
                result = {"error": f"Unknown action: {action}"}
            
            results.append(result)
            
            # Small delay between operations to avoid rate limiting
            time.sleep(0.5)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get operation statistics"""
        return self.stats.copy()
    
    def reset_stats(self) -> None:
        """Reset operation statistics"""
        self.stats = {
            'operations': 0,
            'successful': 0,
            'failed': 0,
            'errors': []
        }


def load_config(config_file: str, wiki_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Load wiki configuration from unified config file
    
    Args:
        config_file: Path to configuration file
        wiki_name: Specific wiki to load (uses default if not specified)
        
    Returns:
        Wiki configuration dictionary
    """
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Handle unified format with 'wikis' structure
    if 'wikis' not in config:
        raise ValueError("Invalid config format: missing 'wikis' section")
    
    if wiki_name:
        if wiki_name not in config["wikis"]:
            raise ValueError(f"Wiki '{wiki_name}' not found in config")
        wiki_config = config["wikis"][wiki_name]
    else:
        # Use default wiki
        default = config.get("default_wiki")
        if not default or default not in config["wikis"]:
            raise ValueError("No valid default wiki specified")
        wiki_config = config["wikis"][default]
    
    return wiki_config


def main():
    """Command-line interface for WikiEditor"""
    parser = argparse.ArgumentParser(
        description="MediaWiki editing client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Edit a page using direct credentials
    %(prog)s --url https://wiki.example.com --user Bot --password pass123 \\
             --action edit --title "Test Page" --text "New content"
    
    # Append to a page using config file
    %(prog)s --config config/config.json --wiki production \\
             --action edit --title "Log Page" --append "\\n* New log entry"
    
    # Move a page
    %(prog)s --config config/config.json --wiki test \\
             --action move --from "Old Title" --to "New Title" --reason "Reorganization"
    
    # Batch operations from file
    %(prog)s --config config/config.json --wiki test --batch operations.json
    
    # Get page content
    %(prog)s --config config/config.json --wiki production \\
             --action get --title "Main Page"
        """
    )
    
    # Connection arguments
    conn_group = parser.add_argument_group('connection')
    conn_group.add_argument('--config', help='Configuration file path')
    conn_group.add_argument('--wiki', help='Wiki name from config file')
    conn_group.add_argument('--url', help='Wiki URL (e.g., https://wiki.example.com)')
    conn_group.add_argument('--user', '--username', dest='username', help='Username')
    conn_group.add_argument('--password', help='Password')
    conn_group.add_argument('--subdomain', help='Subdomain for routing')
    conn_group.add_argument('--insecure', action='store_true', help='Disable SSL verification')
    
    # Action arguments
    action_group = parser.add_argument_group('actions')
    action_group.add_argument('--action', choices=['edit', 'move', 'delete', 'get'],
                             help='Action to perform')
    action_group.add_argument('--batch', help='Batch operations JSON file')
    
    # Edit arguments
    edit_group = parser.add_argument_group('edit options')
    edit_group.add_argument('--title', help='Page title')
    edit_group.add_argument('--text', help='Page text (replaces existing)')
    edit_group.add_argument('--append', help='Text to append')
    edit_group.add_argument('--prepend', help='Text to prepend')
    edit_group.add_argument('--summary', help='Edit summary')
    edit_group.add_argument('--createonly', action='store_true', help='Only create if new')
    edit_group.add_argument('--minor', action='store_true', help='Mark as minor edit')
    edit_group.add_argument('--nobot', action='store_true', help='Don\'t mark as bot edit')
    
    # Move arguments
    move_group = parser.add_argument_group('move options')
    move_group.add_argument('--from', dest='from_title', help='Source page title')
    move_group.add_argument('--to', dest='to_title', help='Destination page title')
    move_group.add_argument('--reason', help='Move/delete reason')
    move_group.add_argument('--noredirect', action='store_true', help='Don\'t create redirect')
    move_group.add_argument('--notalkmove', action='store_true', help='Don\'t move talk page')
    
    # Get arguments
    get_group = parser.add_argument_group('get options')
    get_group.add_argument('--format', choices=['wikitext', 'html'], default='wikitext',
                          help='Content format to retrieve')
    
    # Output options
    parser.add_argument('--output', help='Output file for results')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    if args.config:
        try:
            config = load_config(args.config, args.wiki)
            url = config["url"]
            username = config["username"]
            password = config["password"]
            verify_ssl = config.get("verify_ssl", True)
            subdomain = config.get("subdomain", args.subdomain)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            sys.exit(1)
    elif args.url and args.username and args.password:
        url = args.url
        username = args.username
        password = args.password
        verify_ssl = not args.insecure
        subdomain = args.subdomain
    else:
        parser.error("Must provide either --config or (--url, --user, --password)")
    
    # Create editor instance
    editor = WikiEditor(
        url=url,
        username=username,
        password=password,
        verify_ssl=verify_ssl,
        subdomain=subdomain
    )
    
    # Handle batch operations
    if args.batch:
        try:
            with open(args.batch, 'r') as f:
                operations = json.load(f)
            
            logger.info(f"Executing {len(operations)} batch operations")
            results = editor.batch_operations(operations)
            
            # Save results if output specified
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)
                logger.info(f"Results saved to {args.output}")
            else:
                print(json.dumps(results, indent=2))
            
        except Exception as e:
            logger.error(f"Batch operation failed: {e}")
            sys.exit(1)
    
    # Handle single operations
    elif args.action:
        result = None
        
        if args.action == 'edit':
            if not args.title:
                parser.error("--title required for edit action")
            
            result = editor.edit_page(
                title=args.title,
                text=args.text,
                append=args.append,
                prepend=args.prepend,
                summary=args.summary or "",
                createonly=args.createonly,
                bot=not args.nobot,
                minor=args.minor
            )
            
        elif args.action == 'move':
            if not args.from_title or not args.to_title:
                parser.error("--from and --to required for move action")
            
            result = editor.move_page(
                from_title=args.from_title,
                to_title=args.to_title,
                reason=args.reason or "",
                noredirect=args.noredirect,
                movetalk=not args.notalkmove
            )
            
        elif args.action == 'delete':
            if not args.title:
                parser.error("--title required for delete action")
            
            result = editor.delete_page(
                title=args.title,
                reason=args.reason or ""
            )
            
        elif args.action == 'get':
            if not args.title:
                parser.error("--title required for get action")
            
            content = editor.get_page_content(args.title, args.format)
            if content:
                if args.output:
                    with open(args.output, 'w') as f:
                        f.write(content)
                    logger.info(f"Content saved to {args.output}")
                else:
                    print(content)
            else:
                logger.error("Failed to retrieve page content")
                sys.exit(1)
        
        # Handle result output for edit/move/delete
        if result:
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(result, f, indent=2)
                logger.info(f"Result saved to {args.output}")
            else:
                print(json.dumps(result, indent=2))
            
            # Check for errors
            if "error" in result:
                sys.exit(1)
    
    else:
        parser.error("Must specify either --action or --batch")
    
    # Print statistics
    stats = editor.get_stats()
    if stats['operations'] > 0:
        logger.info(f"Operations: {stats['operations']}, "
                   f"Successful: {stats['successful']}, "
                   f"Failed: {stats['failed']}")


if __name__ == "__main__":
    main()