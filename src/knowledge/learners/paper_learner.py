# Paper Learner
#
# Extracts knowledge from research papers (PDFs).
# Parses sections, abstracts, formulas, and key findings.

from typing import Any, Dict, List

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, PictureDescriptionVlmOptions
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
        
        smolvlm_picture_description = PictureDescriptionVlmOptions(
            repo_id='HuggingFaceTB/SmolVLM-256M-Instruct',
            prompt="Describe the picture in detail. Make sure to include all the details of the picture."
        )
        pipeline_options = PdfPipelineOptions(
            do_formula_enrichment = True,
            do_picture_description = True,
        )

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )
        result = converter.convert(path)        
        markdown_content = doc.document.export_to_markdown()
        print(markdown_content)

        chunks = []        
        chunks.append(KnowledgeChunk(
            content=f"Paper knowledge from {path}",
            chunk_type="text",
            source=path,
            metadata={"learner": "paper"}
        ))
        
        print(f"[PaperLearner] Learned from paper: {path}")
        return chunks
