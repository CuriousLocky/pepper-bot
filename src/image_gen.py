import os
import re
import uuid
import base64
import asyncio
import requests
from io import BytesIO
from pathlib import Path
from PIL import Image
from openai import AsyncOpenAI
from config import Config

async def generate_image(prompt: str, config: Config) -> tuple[bool, str | None, str | None, str]:
    """
    Generates an image using a chat completion model.
    Returns: (success, full_res_base64, resized_base64_data_url, text_content)
    """
    if not config.image_generation.enabled:
        return False, None, None, "Image generation is disabled."

    api_key = config.image_generation.api_key or config.api.key
    base_url = config.image_generation.api_url or config.api.url
    
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    try:
        response = await client.chat.completions.create(
            model=config.image_generation.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=1.0 # Default for creativity
        )
        
        content = response.choices[0].message.content or ""
        
        # Extract image URL from response message
        message = response.choices[0].message
        images = getattr(message, "images", None)
        
        # If not found as attribute, check model_extra (OpenAI Python SDK v1+)
        if images is None and hasattr(message, "model_extra"):
            images = message.model_extra.get("images")
            
        # If still not found, check if message behaves like a dict
        if images is None and isinstance(message, dict):
            images = message.get("images")

        if not images:
            return False, None, None, "Image URL not found in response."

        try:
            # Handle both object (dot notation) and dict (get) access
            # Expected structure: images[0].url OR images[0].image_url.url
            # Or dict equivalents: images[0]['url'] OR images[0]['image_url']['url']
            first_image = images[0]
            
            def get_val(obj, key):
                if isinstance(obj, dict):
                    return obj.get(key)
                return getattr(obj, key, None)

            image_url = get_val(first_image, "url")
            if not image_url:
                inner_image_url = get_val(first_image, "image_url")
                if inner_image_url:
                    image_url = get_val(inner_image_url, "url")
                
            if not image_url:
                 return False, None, None, "Image URL is empty or structure unknown."
                 
        except (IndexError, AttributeError):
            return False, None, None, "Image URL parsing failed."
        
        
        # Download or Process image
        def download_and_process():
            if image_url.startswith("data:"):
                # Handle Data URI
                # Format: data:image/png;base64,....
                try:
                    header, encoded = image_url.split(",", 1)
                    img_data = base64.b64decode(encoded)
                except Exception:
                    raise ValueError("Invalid Data URL format")
            else:
                # Handle regular URL
                resp = requests.get(image_url)
                resp.raise_for_status()
                img_data = resp.content
            
            # Resize for history
            with Image.open(BytesIO(img_data)) as img:
                # Prepare Full Resolution Base64
                full_buffered = BytesIO()
                # Use original format if possible, default to PNG
                save_format = img.format or "PNG"
                img.save(full_buffered, format=save_format)
                full_img_str = base64.b64encode(full_buffered.getvalue()).decode("utf-8")
                # Detect mime type roughly
                mime_type = f"image/{save_format.lower()}"
                full_res_base64 = f"data:{mime_type};base64,{full_img_str}"

                original_size = img.size
                new_size = (
                    int(original_size[0] * config.image_generation.resolution_scale),
                    int(original_size[1] * config.image_generation.resolution_scale)
                )
                resized_img = img.resize(new_size, Image.Resampling.LANCZOS)
                
                # Convert to base64
                buffered = BytesIO()
                resized_img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                base64_url = f"data:image/png;base64,{img_str}"
                
            return full_res_base64, base64_url

        full_res_base64, base64_url = await asyncio.to_thread(download_and_process)
        return True, full_res_base64, base64_url, content

    except Exception as e:
        print(f"Error generating image: {e}")
        return False, None, None, str(e)