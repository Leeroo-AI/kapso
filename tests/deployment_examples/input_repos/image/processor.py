# Image Processor - ORIGINAL INPUT
#
# Uses Pillow for image processing.
# NO deployment files - just core logic.

import base64
import io
from PIL import Image, ImageFilter, ImageOps


class ImageProcessor:
    """Image processor using Pillow."""
    
    def process(self, image_data: str, width: int = None, filters: list = None) -> dict:
        """Process image with options."""
        if "," in image_data:
            image_data = image_data.split(",")[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        original_size = image.size
        
        if width:
            ratio = width / image.size[0]
            image = image.resize((width, int(image.size[1] * ratio)))
        
        if filters:
            for f in filters:
                if f == "grayscale":
                    image = ImageOps.grayscale(image).convert("RGB")
        
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        output_data = base64.b64encode(buffer.getvalue()).decode()
        
        return {
            "image_data": output_data,
            "original_size": original_size,
            "processed_size": image.size,
        }


def predict(inputs: dict) -> dict:
    """Main prediction function."""
    if "image_data" not in inputs:
        return {"error": "Missing image_data"}
    
    processor = ImageProcessor()
    return processor.process(
        image_data=inputs["image_data"],
        width=inputs.get("width"),
        filters=inputs.get("filters"),
    )


if __name__ == "__main__":
    print("Image processor ready")

