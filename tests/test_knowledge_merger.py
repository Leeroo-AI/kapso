# Test: Knowledge Merger
#
# Simple test to merge staging wiki pages into main KG.
# Automatically starts Neo4j and Weaviate Docker containers if not running.
#
# Required environment variables:
#   - OPENAI_API_KEY: For embeddings

import logging
import subprocess
import time
from pathlib import Path

from dotenv import load_dotenv

from src.knowledge.learners import KnowledgeMerger, MergeInput
from src.knowledge.search.kg_graph_search import parse_wiki_directory

# Enable logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

load_dotenv()


# =============================================================================
# Docker Container Management
# =============================================================================

def is_container_running(name: str) -> bool:
    """Check if a Docker container is running."""
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", f"name={name}", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
        )
        return name in result.stdout
    except Exception:
        return False


def start_neo4j():
    """Start Neo4j Docker container if not running."""
    if is_container_running("neo4j"):
        print("Neo4j container already running")
        return True
    
    print("Starting Neo4j container...")
    try:
        # Remove existing stopped container if any
        subprocess.run(["docker", "rm", "-f", "neo4j"], capture_output=True)
        
        # Start new container
        subprocess.run([
            "docker", "run", "-d",
            "--name", "neo4j",
            "-p", "7474:7474",
            "-p", "7687:7687",
            "-e", "NEO4J_AUTH=neo4j/password",
            "neo4j:latest"
        ], check=True)
        
        # Wait for Neo4j to be ready
        print("Waiting for Neo4j to start...")
        for i in range(30):
            time.sleep(2)
            try:
                result = subprocess.run(
                    ["docker", "exec", "neo4j", "neo4j", "status"],
                    capture_output=True,
                    text=True,
                )
                if "running" in result.stdout.lower():
                    print("Neo4j is ready")
                    return True
            except Exception:
                pass
            print(f"  Waiting... ({i+1}/30)")
        
        print("Neo4j may not be fully ready, continuing anyway...")
        return True
        
    except Exception as e:
        print(f"Failed to start Neo4j: {e}")
        return False


def start_weaviate():
    """Start Weaviate Docker container if not running."""
    if is_container_running("weaviate"):
        print("Weaviate container already running")
        return True
    
    print("Starting Weaviate container...")
    try:
        # Remove existing stopped container if any
        subprocess.run(["docker", "rm", "-f", "weaviate"], capture_output=True)
        
        # Start new container
        subprocess.run([
            "docker", "run", "-d",
            "--name", "weaviate",
            "-p", "8081:8080",
            "-p", "50051:50051",
            "-e", "AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true",
            "-e", "PERSISTENCE_DATA_PATH=/var/lib/weaviate",
            "semitechnologies/weaviate:latest"
        ], check=True)
        
        # Wait for Weaviate to be ready
        print("Waiting for Weaviate to start...")
        for i in range(15):
            time.sleep(2)
            try:
                result = subprocess.run(
                    ["curl", "-s", "http://localhost:8081/v1/.well-known/ready"],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    print("Weaviate is ready")
                    return True
            except Exception:
                pass
            print(f"  Waiting... ({i+1}/15)")
        
        print("Weaviate may not be fully ready, continuing anyway...")
        return True
        
    except Exception as e:
        print(f"Failed to start Weaviate: {e}")
        return False


def ensure_docker_services():
    """Ensure Neo4j and Weaviate are running."""
    print("\n" + "=" * 60)
    print("Checking Docker services...")
    print("=" * 60)
    
    neo4j_ok = start_neo4j()
    weaviate_ok = start_weaviate()
    
    if neo4j_ok and weaviate_ok:
        print("\nAll services ready!")
        # Give extra time for services to fully initialize
        time.sleep(3)
    else:
        print("\nWarning: Some services may not be available")
    
    return neo4j_ok and weaviate_ok


def load_staging_pages(staging_dir: Path):
    """Load wiki pages from staging directory."""
    # Find the session subdirectory (e.g., f0233dd9f9b1)
    session_dirs = [d for d in staging_dir.iterdir() if d.is_dir()]
    if not session_dirs:
        print(f"No session directories found in {staging_dir}")
        return []
    
    # Use most recent session
    session_dir = sorted(session_dirs)[-1]
    print(f"Loading from session: {session_dir.name}")
    
    # Parse pages from session directory
    return parse_wiki_directory(session_dir)


def main():
    """Run merger test."""
    # Ensure Docker services are running
    ensure_docker_services()
    
    print("\n" + "=" * 60)
    print("Knowledge Merger Test")
    print("=" * 60)
    
    # Paths
    main_kg_path = Path("data/wikis")
    staging_path = Path("data/wikis/_staging/huggingface_transformers")
    
    # Load staging pages
    print(f"\nLoading pages from: {staging_path}")
    proposed_pages = load_staging_pages(staging_path)
    
    if not proposed_pages:
        print("No pages found in staging directory")
        return
    
    print(f"Found {len(proposed_pages)} pages to merge")
    
    # Show page summary by type
    by_type = {}
    for page in proposed_pages:
        by_type.setdefault(page.page_type, []).append(page)
    
    print("\nPages by type:")
    for page_type, pages in sorted(by_type.items()):
        print(f"  {page_type}: {len(pages)}")
    
    # Create merge input
    merge_input = MergeInput(
        proposed_pages=proposed_pages,
        main_kg_path=main_kg_path,
    )
    
    print(f"\nMain KG path: {merge_input.main_kg_path}")
    print(f"Neo4j URI: {merge_input.neo4j_uri}")
    print(f"Weaviate collection: {merge_input.weaviate_collection}")
    
    # Run merge
    print("\n" + "-" * 60)
    print("Starting merge...")
    print("-" * 60)
    
    merger = KnowledgeMerger()
    result = merger.merge(merge_input)
    
    # Print results
    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    print(f"  Total proposed: {result.total_proposed}")
    print(f"  Created:        {len(result.created)}")
    print(f"  Merged:         {len(result.merged)}")
    print(f"  Errors:         {len(result.errors)}")
    print(f"  Success:        {result.success}")
    
    if result.errors:
        print("\nErrors:")
        for error in result.errors[:5]:
            print(f"  - {error}")
        if len(result.errors) > 5:
            print(f"  ... and {len(result.errors) - 5} more")
    
    # Cleanup
    merger.close()
    
    print("\n" + "=" * 60)
    print("Test complete!")


if __name__ == "__main__":
    main()

