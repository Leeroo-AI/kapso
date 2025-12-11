# Paper Learner
#
# Extracts knowledge from research papers (PDFs).
# Parses sections, abstracts, formulas, and key findings.

import os
from typing import Any, Dict, List

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, PictureDescriptionVlmOptions, PictureDescriptionApiOptions
from docling_core.types.doc.document import PictureDescriptionData

from src.knowledge.learners.base import Learner, KnowledgeChunk
from src.knowledge.learners.factory import register_learner

@register_learner("paper")
class PaperLearner(Learner):
    """
    Learn from research papers and PDFs.
    
    Extracts knowledge from:
    - Abstract and introduction
    - Methodology sections
    - Key findings and conclusions
    - Formulas and algorithms
    
    Input format:
        {"path": "./paper.pdf"} or {"url": "https://arxiv.org/..."}
    """
    
    @property
    def name(self) -> str:
        return "paper"
    
    def learn(self, source_data: Dict[str, Any]) -> List[KnowledgeChunk]:
        """
        Extract knowledge from a research paper.
        
        Args:
            source_data: Dict with "path" (local file) or "url" (remote PDF)
        Returns:
            List of KnowledgeChunk from the paper
        """
        path = source_data.get("path", source_data.get("url", ""))

        pipeline_options = PdfPipelineOptions(
            do_formula_enrichment = True,
            do_picture_description = True,
            picture_description_options=self._create_picture_description_options(),
            enable_remote_services=True,
        )

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )
        result = converter.convert(path)        
        markdown_content = result.document.export_to_markdown()
        
        # TODO: Convert markdown to KG.
        chunks = []        
        chunks.append(KnowledgeChunk(
            content=f"Paper knowledge from {path}",
            chunk_type="text",
            source=path,
            metadata={"learner": "paper"}
        ))
        
        print(f"[PaperLearner] Learned from paper: {path}")
        return chunks

    def _create_picture_description_options(self) -> PictureDescriptionApiOptions:
        """
        Create the picture description options.
        """
        
        image_description_prompt = """
            Describe the picture in details. Make sure to include all the details, for exampel, convert flows and diagrams to text.
            Ignore examples, and details of messy diagrams. Only extract and summarize the main content and idea of the picture. 
            put your description in the following format:
            <image_description>
                Textual description of the picture.
            </image_description>
        """
        # TODO: Add compatibility for other LLM provider APIs.
        return PictureDescriptionApiOptions(
            url="https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
            },
            params=dict(
                model="gpt-4o",
                max_completion_tokens=500,
            ),
            prompt=image_description_prompt,
            timeout=90,
        )

