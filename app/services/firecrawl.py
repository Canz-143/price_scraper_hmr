import os
import json
import asyncio
import re
from urllib.parse import urlparse, parse_qs
import google.generativeai as genai

from crawl4ai import (
    AsyncWebCrawler, 
    CrawlerRunConfig, 
    BrowserConfig,
    CacheMode
)
from pydantic import BaseModel, Field

from app.config import GOOGLE_API_KEY


class ProductPrice(BaseModel):
    combined_price: str = Field(
        ...,
        description="Price with currency symbol (e.g., $2000)"
    )
    price: str = Field(
        ...,
        description="Numeric price without symbol (e.g., 2000)"
    )
    currency_code: str = Field(
        ...,
        description="Currency code (e.g., USD, EUR)"
    )
    website_name: str = Field(
        ...,
        description="Website name"
    )
    product_page_url: str = Field(
        ...,
        description="Direct product page URL"
    )


def is_valid_url(url):
    """Check if URL is valid and has proper format."""
    if not url:
        return False
    
    if not url.startswith(('http://', 'https://')):
        return False
    
    try:
        parsed = urlparse(url)
        if not parsed.netloc:
            return False
        if '.' not in parsed.netloc:
            return False
        if '/blocked?' in url or 'blocked' in parsed.path:
            return False
        return True
    except Exception:
        return False


def is_search_or_collection_page(url):
    """Determine if URL is a search result or collection page."""
    if not url:
        return True
    
    parsed_url = urlparse(url.lower())
    path = parsed_url.path
    query_params = parse_qs(parsed_url.query)
    
    search_indicators = [
        r'/search', r'/results', r'/find', r'/query', r'/s/',
        r'/buscar', r'/recherche', r'/suche',
    ]
    
    collection_indicators = [
        r'/category', r'/categories', r'/collection', r'/collections',
        r'/browse', r'/catalog', r'/products(?:/(?:all|list))?$',
        r'/items', r'/list', r'/archive', r'/tag/', r'/tags/',
        r'/c/', r'/cat/', r'/department', r'/shop(?:/(?:all|category))?$',
    ]
    
    for pattern in search_indicators + collection_indicators:
        if re.search(pattern, path):
            return True
    
    search_params = ['q', 'query', 'search', 'keyword', 'term', 'find', 's', 'k', 'p']
    for param in search_params:
        if param in query_params:
            return True
    
    collection_params = ['category', 'cat', 'collection', 'tag', 'filter', 'sort']
    collection_param_count = sum(1 for param in collection_params if param in query_params)
    
    if collection_param_count >= 2:
        return True
    
    pagination_params = ['page', 'p', 'offset', 'start', 'limit']
    has_pagination = any(param in query_params for param in pagination_params)
    
    if has_pagination and collection_param_count >= 1:
        return True
    
    domain = parsed_url.netloc
    
    if 'amazon.' in domain:
        if '/s?' in url or '/s/' in path:
            return True
        if re.search(r'/b/|/gp/browse/|/departments/', path):
            return True
    elif 'ebay.' in domain:
        if '/sch/' in path or '/b/' in path:
            return True
    elif 'shopify' in domain or '/collections/' in path:
        if '/collections/' in path and not re.search(r'/collections/[^/]+/products/', path):
            return True
    elif 'etsy.' in domain:
        if '/search/' in path or '/c/' in path:
            return True
    elif 'walmart.' in domain:
        if '/search/' in path or '/browse/' in path:
            return True
    elif 'target.' in domain:
        if '/s/' in path or '/c/' in path:
            return True
    
    return False


def is_likely_product_page(url):
    """Check if URL looks like a product page."""
    if not url:
        return False
    
    parsed_url = urlparse(url.lower())
    path = parsed_url.path
    
    product_indicators = [
        r'/product/', r'/item/', r'/p/', r'/dp/',
        r'/itm/', r'/listing/', r'/products/[^/]+$',
        r'/[^/]+-p-\d+', r'/\d+\.html?$',
    ]
    
    for pattern in product_indicators:
        if re.search(pattern, path):
            return True
    
    path_parts = [part for part in path.split('/') if part]
    if path_parts:
        last_part = path_parts[-1]
        if re.search(r'^[a-zA-Z0-9\-_]+$', last_part) and len(last_part) > 3:
            return True
    
    return False


async def extract_price_with_gemini(html: str, url: str, model, verbose=True) -> dict:
    """
    Extract price directly using Gemini API with improved error handling.
    
    Args:
        html: Raw HTML content
        url: Product page URL (used as fallback only)
        model: Configured Gemini model
        verbose: Print debug information
    
    Returns:
        dict: Extracted price data or None if extraction fails
    """
    
    # Simple, focused prompt that works better
    prompt = f"""Extract product pricing information from this HTML page.

Find and return ONLY this information in JSON format:
1. combined_price: The visible price with currency symbol (e.g., "$1,299.99" or "₱2,500")
2. price: Just the number without any symbols (e.g., "1299.99" or "2500")
3. currency_code: Three-letter code (USD, PHP, EUR, GBP, etc.)
4. website_name: Store name (Amazon, eBay, Lazada, etc.)
5. product_page_url: Find canonical URL from these sources in order:
   - <link rel="canonical" href="...">
   - <meta property="og:url" content="...">
   - If not found use: {url}

Look for price in:
- Elements with "price" in class/id
- Meta tags: og:price:amount, product:price
- JSON-LD structured data
- Data attributes: data-price

IMPORTANT: 
- Extract the CURRENT/SALE price (ignore crossed-out or "was" prices)
- For "price" field, remove ALL symbols and separators except decimal point
- Return valid JSON only, no markdown formatting

Example output:
{{
  "combined_price": "$1,299.99",
  "price": "1299.99",
  "currency_code": "USD",
  "website_name": "Amazon",
  "product_page_url": "https://amazon.com/product/12345"
}}

HTML (first 100000 characters):
{html[:100000]}"""
    
    try:
        if verbose:
            print(f"\n🔍 Extracting price for: {url}")
            print(f"   HTML length: {len(html)} chars")
        
        # Generate response with safer settings
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0,
                "top_p": 0.95,
                "max_output_tokens": 1000,
                "response_mime_type": "application/json"  # Force JSON output
            },
            safety_settings={
                'HARASSMENT': 'block_none',
                'HATE_SPEECH': 'block_none', 
                'SEXUALLY_EXPLICIT': 'block_none',
                'DANGEROUS_CONTENT': 'block_none'
            }
        )
        
        # Get the response text
        text = response.text.strip()
        
        if verbose:
            print(f"   Raw response length: {len(text)} chars")
            print(f"   Response preview: {text[:200]}")
        
        # Try to parse JSON
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            if "```json" in text:
                json_match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group(1))
                else:
                    if verbose:
                        print(f"   ❌ Failed to extract JSON from markdown")
                    return None
            elif "```" in text:
                json_match = re.search(r'```\s*(\{.*?\})\s*```', text, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group(1))
                else:
                    if verbose:
                        print(f"   ❌ Failed to extract JSON from code block")
                    return None
            else:
                # Try to find JSON object in the text
                json_match = re.search(r'\{.*\}', text, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group(0))
                else:
                    if verbose:
                        print(f"   ❌ No JSON found in response")
                    return None
        
        # Validate the response has required fields
        if isinstance(data, dict) and "price" in data:
            if verbose:
                print(f"   ✅ Extracted: {data.get('combined_price', 'N/A')}")
            return data
        else:
            if verbose:
                print(f"   ⚠️  Response missing 'price' field")
            return None
        
    except Exception as e:
        if verbose:
            print(f"   ❌ Extraction error: {type(e).__name__}: {str(e)}")
        return None


async def call_crawl4ai_extractor(links, request_id=None, verbose=True):
    """
    SIMPLIFIED APPROACH with better debugging:
    1. Use Crawl4AI ONLY for fetching raw HTML (fast)
    2. Use Gemini API directly for extraction (no Crawl4AI extraction layer)
    
    Args:
        links (list): URLs to crawl
        request_id: Optional request identifier
        verbose: Print debug information
    
    Returns:
        dict: Response with success status and ecommerce_links array
    """
    # Filter and limit links
    limited_links = links[:10]
    filtered_links = []
    
    if verbose:
        print(f"\n📋 Processing {len(limited_links)} links...")
    
    for url in limited_links:
        if not is_valid_url(url):
            if verbose:
                print(f"   ❌ Invalid URL: {url}")
            continue
        if is_search_or_collection_page(url):
            if verbose:
                print(f"   ⏭️  Skipping collection page: {url}")
            continue
        if not is_likely_product_page(url):
            if verbose:
                print(f"   ⏭️  Not a product page: {url}")
            continue
        filtered_links.append(url)
        if verbose:
            print(f"   ✅ Valid product URL: {url}")
    
    if not filtered_links:
        return {
            "success": False,
            "error": "No valid product URLs after filtering.",
            "data": {"ecommerce_links": []}
        }
    
    if verbose:
        print(f"\n🎯 {len(filtered_links)} valid product URLs to crawl\n")
    
    # Configure Gemini
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    # Browser config for speed
    browser_config = BrowserConfig(
        headless=True,
        java_script_enabled=True,
        accept_downloads=False,
    )
    
    # Simple run config - JUST GET THE HTML
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        wait_until="domcontentloaded",
        page_timeout=30000,
        word_count_threshold=10,
        stream=True,
        semaphore_count=3
    )
    
    ecommerce_links = []
    
    # Crawl pages and extract prices
    async with AsyncWebCrawler(config=browser_config, verbose=False) as crawler:
        async for result in await crawler.arun_many(urls=filtered_links, config=run_config):
            if result.success and result.html:
                if verbose:
                    print(f"✅ Crawled: {result.url}")
                
                # Extract price using Gemini
                extracted_data = await extract_price_with_gemini(
                    html=result.html,
                    url=result.url,
                    model=model,
                    verbose=verbose
                )
                
                if extracted_data and isinstance(extracted_data, dict):
                    # Transform to match required format
                    ecommerce_links.append({
                        "website_url": extracted_data.get("product_page_url", result.url),
                        "price_string": extracted_data.get("price", ""),
                        "website_name": extracted_data.get("website_name", ""),
                        "currency_code": extracted_data.get("currency_code", ""),
                        "price_combined": extracted_data.get("combined_price", "")
                    })
                else:
                    if verbose:
                        print(f"   ⚠️  No price extracted for {result.url}\n")
            else:
                if verbose:
                    print(f"❌ Failed to crawl: {result.url}")
                    if result.error_message:
                        print(f"   Error: {result.error_message}\n")
    
    if verbose:
        print(f"\n✨ Extraction complete: {len(ecommerce_links)}/{len(filtered_links)} products with prices\n")
    
    return {
        "success": True,
        "data": {
            "ecommerce_links": ecommerce_links
        }
    }


# For compatibility
call_firecrawl_extractor = call_crawl4ai_extractor
