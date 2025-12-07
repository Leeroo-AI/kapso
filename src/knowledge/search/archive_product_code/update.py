#!/usr/bin/env python3
"""
Wiki Update Pipeline - Updates Neo4j index with latest wiki changes.

This pipeline:
1. Fetches latest changes from MediaWiki using monitor.py
2. Parses wiki content to extract nodes and edges with parser.py
3. Updates Neo4j index with new/modified/deleted pages
4. Maintains state in latest_changes directory

Usage:
    python src/update.py --domain https://acme.leeroo.com --user agent --password PASSWORD
    python src/update.py --config config/config.json --wiki production
    
    # With Neo4j custom settings
    python src/update.py --domain https://acme.leeroo.com --user agent --password PASSWORD \
                         --neo4j-uri bolt://localhost:7687 --neo4j-user neo4j --neo4j-password neo4jpass
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from urllib.parse import urlparse

# Add parent directory to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.monitor import WikiMonitor
from src.parser import WikiParser
from src.indexer import WikiIndexer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class WikiUpdatePipeline:
    """Pipeline for updating Neo4j index with wiki changes"""
    
    def __init__(self, 
                 domain: str,
                 username: str,
                 password: str,
                 neo4j_uri: str = None,
                 neo4j_user: str = None,
                 neo4j_password: str = None,
                 openai_api_key: str = None,
                 use_weaviate: bool = True,
                 weaviate_url: str = "http://localhost:8080"):
        """
        Initialize the update pipeline.
        
        Args:
            domain: Wiki domain (e.g., 'https://acme.leeroo.com')
            username: MediaWiki username
            password: MediaWiki password
            neo4j_uri: Neo4j URI (default from env)
            neo4j_user: Neo4j username (default from env)
            neo4j_password: Neo4j password (default from env)
            openai_api_key: OpenAI API key for embeddings (optional)
            use_weaviate: Whether to use Weaviate for vector storage (default: True)
            weaviate_url: Weaviate server URL
        """
        # Parse domain to extract hostname
        parsed_url = urlparse(domain if domain.startswith('http') else f'https://{domain}')
        self.domain = parsed_url.netloc or parsed_url.path
        self.use_https = parsed_url.scheme == 'https' or parsed_url.scheme == ''
        
        self.username = username
        self.password = password
        
        # Neo4j settings (with defaults from environment)
        self.neo4j_uri = neo4j_uri or os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self.neo4j_user = neo4j_user or os.getenv('NEO4J_USER', 'neo4j')
        self.neo4j_password = neo4j_password or os.getenv('NEO4J_PASSWORD')
        
        # OpenAI settings
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        
        # Weaviate settings
        self.use_weaviate = use_weaviate
        self.weaviate_url = weaviate_url or os.getenv('WEAVIATE_URL', 'http://localhost:8080')
        
        # Set up latest_changes directory structure
        self.base_dir = Path('latest_changes')
        self.base_dir.mkdir(exist_ok=True)
        
        # Create domain-specific directory
        safe_domain = self.domain.replace('/', '_').replace(':', '_').replace('.', '_')
        self.domain_dir = self.base_dir / safe_domain
        self.domain_dir.mkdir(exist_ok=True)
        
        # Generate Neo4j database name from domain
        # Database names in Neo4j must start with letter and contain only alphanumeric and underscore
        self.neo4j_database = f"wiki_{safe_domain}".lower()
        
        # Paths for state and changes
        self.state_path = self.domain_dir / 'state.json'
        self.changes_path = self.domain_dir / 'changes.ndjson'
        self.update_log_path = self.domain_dir / 'update_log.json'
        
        # Initialize components
        self.monitor = None
        self.parser = WikiParser()
        self.indexer = None
        
        # Track statistics for this update
        self.stats = {
            'start_time': None,
            'end_time': None,
            'total_changes': 0,
            'pages_updated': 0,
            'pages_deleted': 0,
            'pages_renamed': 0,
            'pages_indexed': 0,
            'errors': 0,
            'nodes_created': 0,
            'edges_created': 0
        }
        
        logger.info(f"Initialized pipeline for {self.domain}")
        logger.info(f"Working directory: {self.domain_dir}")
        logger.info(f"Neo4j database: {self.neo4j_database}")
    
    def initialize_indexer(self) -> bool:
        """Initialize Neo4j indexer with domain-prefixed IDs"""
        try:
            if not self.neo4j_password:
                logger.error("Neo4j password not provided. Set NEO4J_PASSWORD env var or use --neo4j-password")
                return False
            
            # Initialize indexer with domain for ID prefixing
            self.indexer = WikiIndexer(
                uri=self.neo4j_uri,
                user=self.neo4j_user,
                password=self.neo4j_password,
                domain=self.domain,  # Pass domain for ID prefixing
                database="neo4j",  # Use default database (Community Edition)
                openai_api_key=self.openai_api_key,
                use_weaviate=self.use_weaviate,
                weaviate_url=self.weaviate_url
            )
            
            # Create indexes if they don't exist
            self.indexer.create_indexes()
            
            logger.info(f"Connected to Neo4j at {self.neo4j_uri} (domain: {self.domain})")
            if self.openai_api_key:
                logger.info("OpenAI embeddings enabled")
            if self.use_weaviate:
                logger.info(f"Weaviate vector storage enabled at {self.weaviate_url}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j indexer: {e}")
            return False
    
    def run(self, use_embeddings: bool = True, dry_run: bool = False) -> Dict:
        """
        Run the update pipeline.
        
        Args:
            use_embeddings: Whether to generate embeddings for pages (default: True)
            dry_run: If True, only check changes without updating index
            
        Returns:
            Dictionary with update statistics
        """
        logger.info("=" * 70)
        logger.info(f"🚀 Starting wiki update pipeline for {self.domain}")
        logger.info("=" * 70)
        
        self.stats['start_time'] = datetime.now().isoformat()
        
        # Initialize monitor
        self.monitor = WikiMonitor(
            domain=self.domain,
            username=self.username,
            password=self.password,
            state_path=str(self.state_path),
            output_path=str(self.changes_path),
            include_wikitext=True,  # We need wikitext for parsing
            use_https=self.use_https
        )
        
        # Check for changes
        logger.info("\n📋 Step 1: Checking for changes...")
        changes = self.monitor.check_changes(save=True)
        self.stats['total_changes'] = len(changes)
        
        if not changes:
            logger.info("✅ No new changes detected")
            self.stats['end_time'] = datetime.now().isoformat()
            self._save_update_log()
            return self.stats
        
        logger.info(f"Found {len(changes)} new change(s)")
        
        # If dry run, just display changes and exit
        if dry_run:
            logger.info("\n🔍 DRY RUN - Changes detected but not processed:")
            self.monitor.display_changes(verbose=True)
            self.stats['end_time'] = datetime.now().isoformat()
            return self.stats
        
        # Initialize Neo4j indexer
        logger.info("\n🗄️ Step 2: Connecting to Neo4j...")
        if not self.initialize_indexer():
            logger.error("Failed to connect to Neo4j. Aborting.")
            self.stats['errors'] += 1
            self.stats['end_time'] = datetime.now().isoformat()
            self._save_update_log()
            return self.stats
        
        # Process each change
        logger.info("\n⚡ Step 3: Processing changes...")
        for i, change in enumerate(changes, 1):
            try:
                self._process_change(change, i, len(changes), use_embeddings)
            except Exception as e:
                logger.error(f"Error processing change {i}: {e}")
                self.stats['errors'] += 1
        
        # Get final statistics
        logger.info("\n📊 Step 4: Gathering statistics...")
        if self.indexer:
            graph_stats = self.indexer.get_statistics()
            self.stats['graph_stats'] = graph_stats
            logger.info(f"Graph now contains: {graph_stats['total_nodes']} nodes, {graph_stats['total_edges']} edges")
        
        # Clean up
        if self.indexer:
            self.indexer.close()
        
        self.stats['end_time'] = datetime.now().isoformat()
        self._save_update_log()
        
        # Display summary
        self._display_summary()
        
        return self.stats
    
    def _process_change(self, change: Dict, index: int, total: int, use_embeddings: bool):
        """Process a single change"""
        op_type = change.get('op', 'unknown')
        
        logger.info(f"  [{index}/{total}] Processing {op_type}: {change.get('title', change.get('pageid'))}")
        
        if op_type == 'upsert_latest':
            # Page created or updated
            title = change.get('title')
            wikitext = change.get('wikitext')
            
            if not title or not wikitext:
                logger.warning(f"    ⚠️ Missing title or wikitext for page {change.get('pageid')}")
                self.stats['errors'] += 1
                return
            
            # Parse the page to extract nodes and edges
            try:
                node, edges = self.parser.parse(title, wikitext)
                
                # Index the page
                result = self.indexer.index_page(title, wikitext, use_embeddings)
                
                self.stats['pages_updated'] += 1
                self.stats['pages_indexed'] += 1
                self.stats['nodes_created'] += 1
                self.stats['edges_created'] += len(edges)
                
                logger.info(f"    ✅ Indexed: {title} ({node.type.value}) with {len(edges)} links")
                
            except Exception as e:
                logger.error(f"    ❌ Failed to index {title}: {e}")
                self.stats['errors'] += 1
        
        elif op_type == 'delete':
            # Page deleted
            page_id = change.get('pageid')
            title = change.get('title', f'Page_{page_id}')
            
            try:
                self._delete_from_index(title, page_id)
                self.stats['pages_deleted'] += 1
                logger.info(f"    ✅ Deleted: {title}")
            except Exception as e:
                logger.error(f"    ❌ Failed to delete {title}: {e}")
                self.stats['errors'] += 1
        
        elif op_type == 'rename':
            # Page renamed
            old_title = change.get('old_title')
            new_title = change.get('new_title')
            page_id = change.get('pageid')
            
            try:
                self._rename_in_index(old_title, new_title, page_id)
                self.stats['pages_renamed'] += 1
                logger.info(f"    ✅ Renamed: {old_title} → {new_title}")
            except Exception as e:
                logger.error(f"    ❌ Failed to rename {old_title}: {e}")
                self.stats['errors'] += 1
        
        elif op_type == 'error':
            # Error during change detection
            logger.warning(f"    ⚠️ Error in change: {change.get('error')}")
            self.stats['errors'] += 1
    
    def _delete_from_index(self, title: str, page_id: str):
        """Delete a page from Neo4j index"""
        # Create domain-prefixed ID
        prefixed_id = f"{self.domain}:{title}"
        
        with self.indexer.driver.session(database="neo4j") as session:
            # Delete the node and all its relationships
            session.run("""
                MATCH (p:WikiPage {id: $id})
                DETACH DELETE p
            """, id=prefixed_id)
            
            # Also try with page_id if different
            if title != f'Page_{page_id}':
                prefixed_page_id = f"{self.domain}:Page_{page_id}"
                session.run("""
                    MATCH (p:WikiPage {id: $id})
                    DETACH DELETE p
                """, id=prefixed_page_id)
    
    def _rename_in_index(self, old_title: str, new_title: str, page_id: str):
        """Rename a page in Neo4j index"""
        # Create domain-prefixed IDs
        old_prefixed_id = f"{self.domain}:{old_title}"
        new_prefixed_id = f"{self.domain}:{new_title}"
        
        with self.indexer.driver.session(database="neo4j") as session:
            # Update the node's ID and name
            session.run("""
                MATCH (p:WikiPage {id: $old_id})
                SET p.id = $new_id,
                    p.title = $new_title,
                    p.name = $clean_name,
                    p.previous_title = $old_title
                RETURN p
            """, old_id=old_prefixed_id, new_id=new_prefixed_id, 
                new_title=new_title, old_title=old_title,
                clean_name=new_title.split(':')[-1])  # Clean name without prefix
            
            # Note: Relationships are handled automatically since they reference node IDs
            # No need to update relationship properties separately
    
    def _save_update_log(self):
        """Save update log to file"""
        try:
            # Load existing log
            if self.update_log_path.exists():
                with open(self.update_log_path, 'r') as f:
                    log_data = json.load(f)
            else:
                log_data = {'updates': []}
            
            # Add this update
            log_data['updates'].append(self.stats)
            
            # Keep only last 100 updates
            log_data['updates'] = log_data['updates'][-100:]
            
            # Save log
            with open(self.update_log_path, 'w') as f:
                json.dump(log_data, f, indent=2)
            
        except Exception as e:
            logger.warning(f"Could not save update log: {e}")
    
    def _display_summary(self):
        """Display summary of the update"""
        print("\n" + "=" * 70)
        print("📈 UPDATE SUMMARY")
        print("=" * 70)
        print(f"Domain: {self.domain}")
        print(f"Database: {self.neo4j_database}")
        print(f"Start: {self.stats['start_time']}")
        print(f"End: {self.stats['end_time']}")
        print(f"\nChanges processed: {self.stats['total_changes']}")
        print(f"  • Pages updated: {self.stats['pages_updated']}")
        print(f"  • Pages deleted: {self.stats['pages_deleted']}")
        print(f"  • Pages renamed: {self.stats['pages_renamed']}")
        print(f"  • Errors: {self.stats['errors']}")
        
        if 'graph_stats' in self.stats:
            print(f"\nGraph statistics:")
            print(f"  • Total nodes: {self.stats['graph_stats']['total_nodes']}")
            print(f"  • Total edges: {self.stats['graph_stats']['total_edges']}")
            
            if self.stats['graph_stats'].get('nodes_with_embeddings'):
                print(f"  • Nodes with embeddings: {self.stats['graph_stats']['nodes_with_embeddings']}")
        
        print(f"\nWorking directory: {self.domain_dir}")
        print("=" * 70)


def load_config(config_path: str, wiki_name: Optional[str] = None) -> Dict:
    """Load configuration from file"""
    config_file = Path(config_path)
    if not config_file.exists():
        return {}
    
    with open(config_file) as f:
        config = json.load(f)
    
    # Handle unified format with 'wikis' structure
    if 'wikis' in config:
        if wiki_name and wiki_name in config['wikis']:
            return config['wikis'][wiki_name]
        elif not wiki_name and 'default_wiki' in config:
            default = config.get('default_wiki')
            if default in config['wikis']:
                return config['wikis'][default]
    
    return {}


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description='Update Neo4j index with latest wiki changes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using command-line arguments
  %(prog)s --domain https://acme.leeroo.com --user agent --password pass123
  
  # Using config file
  %(prog)s --config config/config.json --wiki production
  
  # With Neo4j custom settings
  %(prog)s --domain https://acme.leeroo.com --user agent --password pass123 \\
      --neo4j-uri bolt://localhost:7687 --neo4j-user neo4j --neo4j-password neo4jpass
  
  # Disable embeddings (embeddings enabled by default)
  %(prog)s --domain https://acme.leeroo.com --user agent --password pass123 --no-embeddings
  
  # Disable Weaviate (Weaviate enabled by default)
  %(prog)s --domain https://acme.leeroo.com --user agent --password pass123 --no-weaviate
  
  # Dry run (check changes without updating)
  %(prog)s --domain https://acme.leeroo.com --user agent --password pass123 --dry-run
        """
    )
    
    # Config file option
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--wiki', help='Wiki name from config file')
    
    # Wiki credentials
    parser.add_argument('--domain', help='Wiki domain (e.g., https://acme.leeroo.com)')
    parser.add_argument('--user', help='MediaWiki username')
    parser.add_argument('--password', help='MediaWiki password')
    
    # Neo4j settings
    parser.add_argument('--neo4j-uri', help='Neo4j connection URI')
    parser.add_argument('--neo4j-user', help='Neo4j username')
    parser.add_argument('--neo4j-password', help='Neo4j password')
    
    # Optional features (embeddings and Weaviate enabled by default)
    parser.add_argument('--no-embeddings', action='store_true', 
                       help='Disable embedding generation (enabled by default, requires OPENAI_API_KEY)')
    parser.add_argument('--no-weaviate', action='store_true',
                       help='Disable Weaviate vector storage (enabled by default)')
    parser.add_argument('--weaviate-url', help='Weaviate server URL (default: http://localhost:8080)')
    
    # Operational flags
    parser.add_argument('--dry-run', action='store_true',
                       help='Check changes without updating index')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed output')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration if provided
    config = {}
    if args.config:
        config = load_config(args.config, args.wiki)
        if config and not args.domain:
            print(f"📖 Using {'wiki: ' + args.wiki if args.wiki else 'default'} from {args.config}")
    
    # Determine credentials (command-line overrides config)
    domain = args.domain or config.get('domain')
    username = args.user or config.get('username')
    password = args.password or config.get('password')
    
    # Validate required fields
    if not all([domain, username, password]):
        print("❌ Error: Missing required configuration")
        print("\nRequired: domain, username, password")
        print("\nProvide via:")
        print("  1. Command line: --domain --user --password")
        print("  2. Config file: --config config/config.json")
        sys.exit(1)
    
    # Create and run pipeline
    pipeline = WikiUpdatePipeline(
        domain=domain,
        username=username,
        password=password,
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        use_weaviate=not args.no_weaviate,  # Enabled by default unless --no-weaviate
        weaviate_url=args.weaviate_url
    )
    
    # Run the update
    stats = pipeline.run(
        use_embeddings=not args.no_embeddings,  # Enabled by default unless --no-embeddings
        dry_run=args.dry_run
    )
    
    # Exit with appropriate code
    if stats['errors'] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()