"""
VLM-based chart data extraction module.

This file contains the baseline implementation for extracting tabular data
from chart images using a Vision Language Model (VLM).

Kapso will optimize this to improve extraction accuracy.
"""

import base64
from pathlib import Path
from openai import OpenAI


def image_to_data_uri(image_path: Path) -> str:
    """
    Convert an image file to a data URI for the OpenAI API.
    
    Args:
        image_path: Path to the image file.
    
    Returns:
        A data URI string suitable for the OpenAI vision API.
    """
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    
    # Determine MIME type from extension
    suffix = image_path.suffix.lower()
    mime_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    mime_type = mime_types.get(suffix, "image/png")
    
    return f"data:{mime_type};base64,{data}"


def build_prompt() -> str:
    """
    Build the prompt template for chart data extraction.
    
    This prompt instructs the VLM to extract tabular data from chart images.
    Kapso will optimize this prompt to improve accuracy.
    
    Returns:
        The prompt string.
    """
    return (
        "You are a precise data extraction model. Given a chart image, extract the underlying data table.\n"
        "Return ONLY the CSV text with a header row and no markdown code fences.\n"
        "Rules:\n"
        "- The first column must be the x-axis values with its exact axis label as the header.\n"
        "- Include one column per data series using the legend labels as headers.\n"
        "- Preserve the original order of x-axis ticks as they appear.\n"
        "- Use plain CSV (comma-separated), no explanations, no extra text.\n"
    )


def clean_to_csv(text: str) -> str:
    """
    Clean the VLM output to extract pure CSV content.
    
    Removes markdown code fences and extra whitespace.
    
    Args:
        text: Raw VLM output text.
    
    Returns:
        Cleaned CSV string.
    """
    # Remove markdown code fences if present
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```csv or ```)
        lines = lines[1:]
        # Remove last line if it's closing fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    
    return text.strip()


class VLMExtractor:
    """
    Vision Language Model extractor for chart data.
    
    Uses OpenAI's vision-capable models to extract tabular data from chart images.
    """
    
    def __init__(self, model: str = "gpt-4o-mini"):
        """
        Initialize the VLM extractor.
        
        Args:
            model: The OpenAI model to use for extraction.
        """
        self.client = OpenAI()
        self.model = model
    
    def image_to_csv(self, image_path: Path) -> str:
        """
        Extract tabular data from a chart image.
        
        Args:
            image_path: Path to the chart image file.
        
        Returns:
            CSV string containing the extracted data.
        """
        prompt = build_prompt()
        image_uri = image_to_data_uri(image_path)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_uri}}
                    ],
                }
            ],
        )
        
        text = response.choices[0].message.content or ""
        return clean_to_csv(text)
