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


async def extract_price_with_gemini(html: str, url: str, model) -> dict:
    """
    Extract price directly using Gemini API (no Crawl4AI extraction layer).
    
    Args:
        html: Raw HTML content
        url: Product page URL (used as fallback only)
        model: Configured Gemini model
    
    Returns:
        dict: Extracted price data or None if extraction fails
    """
    
    prompt = f"""You are analyzing RAW HTML to extract precise product pricing information.

CRITICAL: You're receiving the COMPLETE HTML source code. Look for price information in:
- HTML elements with classes/ids containing: 'price', 'cost', 'amount', 'value'
- Meta tags: <meta property='og:price:amount'>, <meta itemprop='price'>
- Schema.org markup: <span itemprop='price'>, JSON-LD scripts
- Data attributes: data-price, data-amount, data-cost
- JavaScript variables: window.price, dataLayer, product objects

EXTRACTION RULES:
1. combined_price: Extract the EXACT price text as displayed (e.g., '$2,499.99', '₱2,500.00')
2. price: Extract ONLY the numeric value without symbols or separators (e.g., '2499.99', '2500.00')
   - Remove ALL currency symbols: $, €, ₱, £, ¥, Rs
   - Remove thousand separators (commas, spaces, periods used as thousands)
   - Keep ONLY the decimal point
3. currency_code: 3-letter ISO code (USD, PHP, EUR, GBP, etc.)
4. website_name: E-commerce site name (Amazon, eBay, Lazada, etc.)
5. product_page_url: Extract the canonical product URL from the HTML itself
   - Check <link rel="canonical"> tag
   - Check <meta property="og:url"> tag
   - Check window.location or JavaScript variables
   - If not found, use the current page URL as fallback

SEARCH PRIORITY FOR PRODUCT URL:
1. <link rel="canonical" href="..."> - most reliable
2. <meta property="og:url" content="..."> - Open Graph URL
3. <meta property="product:url" content="..."> - Product-specific meta tag
4. JSON-LD structured data with @type "Product" and "url" field
5. If none found, use: {url}

SEARCH PRIORITY FOR PRICE:
1. Check structured data (JSON-LD, Schema.org, Open Graph)
2. Check common price element patterns
3. Look for CURRENT/SALE price (ignore crossed-out prices)
4. Choose price near 'Add to Cart' button if multiple exist

Return ONLY a valid JSON object in this exact format:
{{
    "combined_price": "exact price string",
    "price": "numeric value only",
    "currency_code": "ISO code",
    "website_name": "site name",
    "product_page_url": "canonical URL from HTML"
}}

If no price found, return all fields as empty strings.

HTML to analyze:
{html[:50000]}"""  # Limit HTML to ~50k chars to stay within token limits
    
    try:
        # Generate response
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0,
                "max_output_tokens": 500,  # We only need a small JSON response
            }
        )
        
        # Extract JSON from response
        text = response.text.strip()
        
        # Remove markdown code blocks if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        
        # Parse JSON
        data = json.loads(text.strip())
        
        # Validate it has the required fields
        if isinstance(data, dict) and "price" in data:
            return data
        
        return None
        
    except Exception as e:
        print(f"Gemini extraction error for {url}: {e}")
        return None


async def call_crawl4ai_extractor(links, request_id=None):
    """
    SIMPLIFIED APPROACH: 
    1. Use Crawl4AI ONLY for fetching raw HTML (fast)
    2. Use Gemini API directly for extraction (no Crawl4AI extraction layer)
    
    Benefits:
    - Faster (no double processing)
    - More control over prompts
    - Easier to debug
    - Lower overhead
    
    Args:
        links (list): URLs to crawl
        request_id: Optional request identifier
    
    Returns:
        dict: Response with success status and ecommerce_links array
    """
    # Filter and limit links
    limited_links = links[:10]
    filtered_links = []
    
    for url in limited_links:
        if not is_valid_url(url):
            continue
        if is_search_or_collection_page(url):
            continue
        if not is_likely_product_page(url):
            continue
        filtered_links.append(url)
    
    if not filtered_links:
        return {
            "success": False,
            "error": "No valid product URLs after filtering.",
            "data": {"ecommerce_links": []}
        }
    
    # Configure Gemini
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    # Browser config for speed
    browser_config = BrowserConfig(
        headless=True,
        java_script_enabled=True,
        accept_downloads=False,
    )
    
    # Simple run config - JUST GET THE HTML, NO EXTRACTION
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        wait_until="domcontentloaded",
        page_timeout=30000,
        word_count_threshold=10,
        stream=True,
        semaphore_count=3
        # NO extraction_strategy here!
    )
    
    ecommerce_links = []
    
    # Step 1: Crawl pages to get raw HTML (Crawl4AI's job)
    async with AsyncWebCrawler(config=browser_config, verbose=True) as crawler:
        async for result in await crawler.arun_many(urls=filtered_links, config=run_config):
            if result.success and result.html:
                # Step 2: Extract price using Gemini directly (no Crawl4AI layer)
                extracted_data = await extract_price_with_gemini(
                    html=result.html,
                    url=result.url,
                    model=model
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
    
    return {
        "success": True,
        "data": {
            "ecommerce_links": ecommerce_links
        }
    }


# For compatibility
call_firecrawl_extractor = call_crawl4ai_extractor
