import os
import re
import asyncio
from google import genai
from google.genai.types import Tool, GoogleSearch
from google.genai.types import GenerateContentConfig
from app.config import GOOGLE_API_KEY

client = genai.Client()
model_id = "gemini-2.0-flash"
search_tool = Tool(google_search=GoogleSearch())

# Add semaphore for Gemini API rate limiting
gemini_semaphore = asyncio.Semaphore(10)  # Limit to 10 concurrent Gemini requests

config = GenerateContentConfig(
    system_instruction=(
        "You are a product identification assistant that can recognize products in images "
        "and provide detailed information about them. Focus on identifying the exact brand, "
        "make, model, and key specs, so users can find and buy this specific item online. "
        "At the end, always clearly list typical search terms either quoted or bullet-listed."
    ),
    tools=[search_tool],
    response_modalities=["TEXT"],
    temperature=0.2,
)

async def analyze_image_async(img_base64: str):
    """Async version with rate limiting"""
    async with gemini_semaphore:
        prompt = (
            "Identify this product with maximum specific detail. "
            "Provide brand, make, model, full product name, and key technical specifications. "
            "At the end, clearly list typical search terms, inside double quotes if possible, "
            "or as bullet points if that's clearer."
        )
        
        # Add small delay to avoid burst limits
        await asyncio.sleep(0.1)
        
        response = client.models.generate_content(
            model=model_id,
            contents=[
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": img_base64
                            }
                        }
                    ]
                }
            ]
        )
        return response.text

# Keep sync version for backward compatibility
def analyze_image(img_base64: str):
    """Synchronous wrapper - consider migrating to async version"""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(analyze_image_async(img_base64))

async def analyze_images_async(img_base64_list):
    """Async version with rate limiting"""
    async with gemini_semaphore:
        prompt = (
            "Identify this product with maximum specific detail. "
            "Provide brand, make, model, full product name, and key technical specifications. "
            "At the end, clearly list typical search terms, inside double quotes if possible, "
            "or as bullet points if that's clearer."
        )
        contents = [{
            "parts": [
                {"text": prompt},
                *[
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": img_base64
                        }
                    }
                    for img_base64 in img_base64_list
                ]
            ]
        }]
        
        # Add small delay to avoid burst limits
        await asyncio.sleep(0.1)
        
        response = client.models.generate_content(
            model=model_id,
            contents=contents,
            config=config
        )
        return response.text

def analyze_images(img_base64_list):
    """Synchronous wrapper - consider migrating to async version"""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(analyze_images_async(img_base64_list))

def extract_search_terms(product_description):
    match = re.search(r'(search terms.*?)[:：]\s*(.*?)$', product_description, re.IGNORECASE | re.DOTALL)
    if not match:
        return []
    block = match.group(2).strip()
    terms = re.findall(r'"(.*?)"', block)
    if terms:
        return terms
    lines = [line.strip('-*• ').strip() for line in block.splitlines() if line.strip()]
    return lines

async def generate_search_variations(base_term):
    """Generate multiple search query variations"""
    variations = [
        base_term,
        f"{base_term} buy online",
        f"{base_term} price Philippines",
        f"{base_term} shop",
        f"{base_term} store"
    ]
    return variations

async def find_shopping_links_async(product_description: str):
    """Async version with proper rate limiting and search variations"""
    search_terms = extract_search_terms(product_description)
    if not search_terms:
        search_terms = [product_description]
    
    # Generate variations for the top 2 search terms
    all_search_queries = []
    for term in search_terms[:2]:
        variations = await generate_search_variations(term)
        all_search_queries.extend(variations)
    
    # Limit total queries to prevent excessive API calls
    all_search_queries = all_search_queries[:8]

    all_links = set()
    
    # Process search terms with rate limiting
    async def process_search_term(term):
        async with gemini_semaphore:
            prompt = f"""
Find direct product purchase pages for: {term}
Use Google Search to locate specific product pages where customers can directly buy this exact item. If possible, look for merchant sites from the Philippines, but if there's none then use global merchant sites. 

Include only:
- Direct product pages with purchase option.
- Sites selling the specific product. 

Exclude:
- Product manuals or documentation. 
- Search results or category pages. 
- Review sites without purchase links.

Output format: URL
"""
            # Add delay to prevent burst limits
            await asyncio.sleep(0.2)

            # Log the term and prompt
            #print(f"[Gemini] Searching for term: {term}")
            #print(f"[Gemini] Prompt sent:\n{prompt}")

            response = client.models.generate_content(
                model=model_id,
                contents=prompt,
                config=config
            )
            
            term_links = set()
            candidate = response.candidates[0]
            if (
                hasattr(candidate, "grounding_metadata")
                and candidate.grounding_metadata
                and hasattr(candidate.grounding_metadata, "grounding_chunks")
                and candidate.grounding_metadata.grounding_chunks
            ):
                for chunk in candidate.grounding_metadata.grounding_chunks:
                    if hasattr(chunk, "web"):
                        term_links.add(chunk.web.uri)
            return term_links
    
    # Process all search terms concurrently but with rate limiting
    results = await asyncio.gather(*[process_search_term(term) for term in all_search_queries])
    
    # Combine all results
    for term_links in results:
        all_links.update(term_links)
    
    # Return up to 10 unique links
    return list(all_links)[:10]

def find_shopping_links(product_description: str):
    """Synchronous wrapper - consider migrating to async version"""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(find_shopping_links_async(product_description))

def extract_shopping_links_urls(response):
    """
    Extracts only the URLs (for Firecrawl) from the Gemini response.
    """
    links = []
    # Try grounding metadata
    if hasattr(response.candidates[0], 'grounding_metadata') and response.candidates[0].grounding_metadata:
        for chunk in response.candidates[0].grounding_metadata.grounding_chunks:
            if hasattr(chunk, 'web'):
                links.append(chunk.web.uri)
    # Fallback
    if not links:
        fallback_links = re.findall(r'\[(.*?)\]\((https?://[^\)]+)\)', response.text)
        links.extend([url for _, url in fallback_links])
    return links
