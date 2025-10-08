import os
import json
import asyncio
import nest_asyncio
import re
from urllib.parse import urlparse, parse_qs

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, LLMConfig, CacheMode
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from pydantic import BaseModel, Field

from app.config import GOOGLE_API_KEY


# nest_asyncio.apply()


class ProductPrice(BaseModel):
    combined_price: str = Field(
        ...,
        description="The product price as a single string, including the currency symbol (e.g., $2000)."
    )
    price: str = Field(
        ...,
        description="The numerical price as a string, without the currency symbol (e.g., 2000)."
    )
    currency_code: str = Field(
        ...,
        description="The currency code (e.g., USD, EUR)."
    )
    website_name: str = Field(
        ...,
        description="The name of the website."
    )
    product_page_url: str = Field(
        ...,
        description="The direct URL to the product page."
    )


def is_valid_url(url):
    """
    Check if URL is valid and has proper format for Firecrawl.
    
    Returns:
        bool: True if valid, False if should be filtered out.
    """
    if not url:
        return False
    
    # Must start with http:// or https://
    if not url.startswith(('http://', 'https://')):
        return False
    
    try:
        parsed = urlparse(url)
        # Must have a valid domain
        if not parsed.netloc:
            return False
        # Must have a proper TLD (at least one dot in domain)
        if '.' not in parsed.netloc:
            return False
        # Filter out blocked/tracking URLs
        if '/blocked?' in url or 'blocked' in parsed.path:
            return False
        return True
    except Exception:
        return False


def is_search_or_collection_page(url):
    """
    Determine if a URL is a search result page or collection page.
    
    Returns:
        bool: True if it should be filtered out, False if it's a product page.
    """
    if not url:
        return True
    
    parsed_url = urlparse(url.lower())
    path = parsed_url.path
    query_params = parse_qs(parsed_url.query)
    
    # Search page indicators
    search_indicators = [
        # URL path patterns
        r'/search',
        r'/results',
        r'/find',
        r'/query',
        r'/s/',
        r'/buscar',      # Spanish
        r'/recherche',   # French
        r'/suche',       # German
    ]
    
    # Collection/category page indicators
    collection_indicators = [
        r'/category',
        r'/categories',
        r'/collection',
        r'/collections',
        r'/browse',
        r'/catalog',
        r'/products(?:/(?:all|list))?$',  # /products, /products/all, /products/list
        r'/items',
        r'/list',
        r'/archive',
        r'/tag/',
        r'/tags/',
        r'/c/',
        r'/cat/',
        r'/department',
        r'/shop(?:/(?:all|category))?$',  # /shop, /shop/all, /shop/category
    ]
    
    # Check URL path for search patterns
    for pattern in search_indicators:
        if re.search(pattern, path):
            return True
    
    # Check URL path for collection patterns
    for pattern in collection_indicators:
        if re.search(pattern, path):
            return True
    
    # Check query parameters for search indicators
    search_params = ['q', 'query', 'search', 'keyword', 'term', 'find', 's', 'k', 'p']
    for param in search_params:
        if param in query_params:
            return True
    
    # Check query parameters for collection/filtering indicators
    collection_params = ['category', 'cat', 'collection', 'tag', 'filter', 'sort']
    collection_param_count = sum(1 for param in collection_params if param in query_params)
    
    # If multiple collection parameters are present, it's likely a collection page
    if collection_param_count >= 2:
        return True
    
    # Check for pagination parameters combined with other indicators
    pagination_params = ['page', 'p', 'offset', 'start', 'limit']
    has_pagination = any(param in query_params for param in pagination_params)
    
    if has_pagination and collection_param_count >= 1:
        return True
    
    # Domain-specific patterns
    domain = parsed_url.netloc
    
    # Amazon-specific patterns
    if 'amazon.' in domain:
        # Amazon search results
        if '/s?' in url or '/s/' in path:
            return True
        # Amazon category pages
        if re.search(r'/b/|/gp/browse/|/departments/', path):
            return True
    
    # eBay-specific patterns
    elif 'ebay.' in domain:
        if '/sch/' in path or '/b/' in path:
            return True
    
    # Shopify stores
    elif 'shopify' in domain or '/collections/' in path:
        if '/collections/' in path and not re.search(r'/collections/[^/]+/products/', path):
            return True
    
    # Etsy-specific patterns
    elif 'etsy.' in domain:
        if '/search/' in path or '/c/' in path:
            return True
    
    # Walmart-specific patterns
    elif 'walmart.' in domain:
        if '/search/' in path or '/browse/' in path:
            return True
    
    # Target-specific patterns
    elif 'target.' in domain:
        if '/s/' in path or '/c/' in path:
            return True
    
    return False


def is_likely_product_page(url):
    """
    Additional check to identify likely product pages.
    
    Returns:
        bool: True if it looks like a product page.
    """
    if not url:
        return False
    
    parsed_url = urlparse(url.lower())
    path = parsed_url.path
    
    # Product page indicators
    product_indicators = [
        r'/product/',
        r'/item/',
        r'/p/',
        r'/dp/',                  # Amazon
        r'/itm/',                 # eBay
        r'/listing/',             # Etsy
        r'/products/[^/]+$',      # Shopify pattern
        r'/[^/]+-p-\d+',          # Common product ID patterns
        r'/\d+\.html?$',          # Numeric product IDs
    ]
    
    for pattern in product_indicators:
        if re.search(pattern, path):
            return True
    
    # Check if path ends with what looks like a product identifier
    path_parts = [part for part in path.split('/') if part]
    if path_parts:
        last_part = path_parts[-1]
        # Product pages often end with product names or IDs
        if re.search(r'^[a-zA-Z0-9\-_]+$', last_part) and len(last_part) > 3:
            return True
    
    return False


async def call_crawl4ai_extractor(links, request_id=None):
    """
    Extract product information from URLs using Crawl4AI and Gemini.
    
    Args:
        links (list): List of URLs to crawl.
        request_id: Optional request identifier for logging.
    
    Returns:
        dict: Response with 'success' status and 'data' containing all results.
    """
    # Filter links
    limited_links = links[:5]
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
            "data": []
        }
    
    # Set Google API key for Gemini
    os.environ['GEMINI_API_KEY'] = GOOGLE_API_KEY
    api_token = os.getenv('GEMINI_API_KEY')
    
    extraction_strategy = LLMExtractionStrategy(
        llm_config=LLMConfig(
            provider="gemini/gemini-2.5-flash",
            api_token=api_token
        ),
        schema=ProductPrice.model_json_schema(),
        extraction_type="schema",
        instruction=(
            "Extract the main product price as a combined string, the price, "
            "the currency code, the website name, and the direct product page URL."
        )
    )
    
    config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        extraction_strategy=extraction_strategy,
    )
    
    async with AsyncWebCrawler(verbose=True) as crawler:
        results = await crawler.arun_many(urls=filtered_links, config=config)
    
    output = []
    for result in results:
        if result.success:
            try:
                extracted_data = json.loads(result.extracted_content)
                output.append({
                    "url": result.url,
                    "product_info": extracted_data,
                    "success": True
                })
            except json.JSONDecodeError:
                output.append({
                    "url": result.url,
                    "error": "Error decoding JSON",
                    "content": result.extracted_content,
                    "success": False
                })
        else:
            output.append({
                "url": result.url,
                "error": result.error_message,
                "success": False
            })
    
    return {
        "success": True,
        "data": output
    }


# For compatibility, alias the function name
call_firecrawl_extractor = call_crawl4ai_extractor
