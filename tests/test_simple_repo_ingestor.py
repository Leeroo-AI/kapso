from src.knowledge.learners import KnowledgePipeline, Source
from dotenv import load_dotenv
import logging

# Enable verbose logging to see what's happening
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

load_dotenv()

# Run full pipeline with unsloth
pipeline = KnowledgePipeline(wiki_dir="data/wikis_test")
result = pipeline.run(Source.Repo("https://github.com/unslothai/unsloth"))

print(f"\n{'='*60}")
print(f"Pages created: {result.total_pages_extracted}")
print(f"Success: {result.success}")
if result.errors:
    print(f"Errors: {result.errors}")