# Test: Repository Ingestor with Claude Code via AWS Bedrock
#
# This test uses Claude Code configured with AWS Bedrock for the LLM backend.
#
# Required environment variables for Bedrock mode:
#   - AWS_REGION: AWS region (e.g., "us-east-1")
#   - One of:
#     - AWS_BEARER_TOKEN_BEDROCK: Bedrock API key (simplest)
#     - AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY: IAM access keys
#     - AWS_PROFILE: SSO profile name (after running 'aws sso login')

from src.knowledge.learners import KnowledgePipeline, Source
from dotenv import load_dotenv
import logging

# Enable verbose logging to see what's happening
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

load_dotenv()

# Configure the pipeline to use Claude Code with AWS Bedrock
# These ingestor_params are passed through to the RepoIngestor,
# which then configures the Claude Code agent with Bedrock settings.
ingestor_params = {
    "use_bedrock": True,           # Use AWS Bedrock instead of direct Anthropic API
    "aws_region": "us-east-1",     # AWS region for Bedrock (adjust as needed)
    "timeout": 1800,               # 30 minutes timeout per phase
    # model: defaults to "us.anthropic.claude-opus-4-5-20251101-v1:0" when use_bedrock=True
    # You can override with any Bedrock-compatible model ID if needed
}

# Run full pipeline with unsloth repo using AWS Bedrock
pipeline = KnowledgePipeline(
    wiki_dir="data/wikis_test",
    ingestor_params=ingestor_params,
)

result = pipeline.run(Source.Repo("https://github.com/unslothai/unsloth"), skip_merge=True)

print(f"\n{'='*60}")
print(f"Pages created: {result.total_pages_extracted}")
print(f"Success: {result.success}")
if result.errors:
    print(f"Errors: {result.errors}")
