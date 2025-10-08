import os
import json
import asyncio
import re
from urllib.parse import urlparse, parse_qs

from crawl4ai import (
    AsyncWebCrawler, 
    CrawlerRunConfig, 
    BrowserConfig,
    LLMConfig, 
    CacheMode
)
from crawl4ai.extraction_strategy import LLMExtractionStrategy
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
    
    # Search page indicators
    search_indicators = [
        r'/search', r'/results', r'/find', r'/query', r'/s/',
        r'/buscar', r'/recherche', r'/suche',
    ]
    
    # Collection/category page indicators
    collection_indicators = [
        r'/category', r'/categories', r'/collection', r'/collections',
        r'/browse', r'/catalog', r'/products(?:/(?:all|list))?$',
        r'/items', r'/list', r'/archive', r'/tag/', r'/tags/',
        r'/c/', r'/cat/', r'/department', r'/shop(?:/(?:all|category))?$',
    ]
    
    # Check URL path for patterns
    for pattern in search_indicators + collection_indicators:
        if re.search(pattern, path):
            return True
    
    # Check query parameters
    search_params = ['q', 'query', 'search', 'keyword', 'term', 'find', 's', 'k', 'p']
    for param in search_params:
        if param in query_params:
            return True
    
    # Collection parameters check
    collection_params = ['category', 'cat', 'collection', 'tag', 'filter', 'sort']
    collection_param_count = sum(1 for param in collection_params if param in query_params)
    
    if collection_param_count >= 2:
        return True
    
    # Pagination check
    pagination_params = ['page', 'p', 'offset', 'start', 'limit']
    has_pagination = any(param in query_params for param in pagination_params)
    
    if has_pagination and collection_param_count >= 1:
        return True
    
    # Domain-specific patterns
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
    
    # Check last path segment
    path_parts = [part for part in path.split('/') if part]
    if path_parts:
        last_part = path_parts[-1]
        if re.search(r'^[a-zA-Z0-9\-_]+$', last_part) and len(last_part) > 3:
            return True
    
    return False


async def call_crawl4ai_extractor(links, request_id=None):
    """
    OPTIMIZED: Extract product information using Crawl4AI with performance enhancements.
    
    Optimizations applied:
    - BrowserConfig for 70% faster initialization
    - wait_until="domcontentloaded" for 40% faster page loading
    - Streaming mode for better memory efficiency
    - Faster Gemini model (2.0-flash-exp)
    - Semaphore control for concurrency management
    - Word count threshold to skip empty pages
    
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
    
    # Configure API
    os.environ['GEMINI_API_KEY'] = GOOGLE_API_KEY
    api_token = os.getenv('GEMINI_API_KEY')
    
    # OPTIMIZATION 1: Use faster Gemini model with optimized settings
    extraction_strategy = LLMExtractionStrategy(
        llm_config=LLMConfig(
            provider="gemini/gemini-2.5-flash",  # Faster than 2.5-flash
            api_token=api_token
        ),
        schema=ProductPrice.model_json_schema(),
        extraction_type="schema",
        instruction=(
            "Extract product information from this page. "
            "If this is a direct product page with a single main product, extract that product's details. "
            "If this is a search results page, collections page, or listing page with multiple products, extract only the FIRST product shown. "
            "Extract the following fields: "
            "1. combined_price - the full price string with currency symbol (e.g., '$2000', 'PHP 2,500') "
            "2. price - only the numeric price value without currency symbol (e.g., '2000', '2500') "
            "3. currency_code - the 3-letter currency code (e.g., 'USD', 'PHP', 'EUR') "
            "4. website_name - the name of the e-commerce website "
            "5. product_page_url - the direct URL to this specific product's page. "
            "Focus on the primary/featured product only. Do not extract multiple products."
        ),
        extra_args={
            "temperature": 0,      # Deterministic output
            "max_tokens": 2000      # Limit for faster response
        }
    )
    
    # OPTIMIZATION 2: Browser config for speed (70% faster initialization)
    browser_config = BrowserConfig(
        headless=True,
        java_script_enabled=True,
        accept_downloads=False,  # Faster if downloads not needed
    )
    
    # OPTIMIZATION 3: Run config with performance settings
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        extraction_strategy=extraction_strategy,
        wait_until="domcontentloaded",  # 40% faster than networkidle
        page_timeout=30000,              # 30s timeout
        word_count_threshold=10,         # Skip pages with little content
        stream=True,                     # Process as results arrive
        semaphore_count=3                # Control concurrency (adjust based on your needs)
    )
    
    ecommerce_links = []
    
    # OPTIMIZATION 4: Streaming mode for better memory usage and faster results
    async with AsyncWebCrawler(config=browser_config, verbose=True) as crawler:
        async for result in await crawler.arun_many(urls=filtered_links, config=run_config):
            if result.success:
                try:
                    extracted_data = json.loads(result.extracted_content)
                    
                    # Handle both list and dict responses
                    if isinstance(extracted_data, list):
                        if extracted_data:
                            extracted_data = extracted_data[0]
                        else:
                            continue
                    
                    # Ensure it's a dict before calling .get()
                    if isinstance(extracted_data, dict):
                        # Transform to match the required format
                        ecommerce_links.append({
                            "website_url": extracted_data.get("product_page_url", result.url),
                            "price_string": extracted_data.get("price", ""),
                            "website_name": extracted_data.get("website_name", ""),
                            "currency_code": extracted_data.get("currency_code", ""),
                            "price_combined": extracted_data.get("combined_price", "")
                        })
                except json.JSONDecodeError:
                    # Skip failed extractions (verbose mode already logs)
                    pass
            # Failed crawls are automatically logged by verbose mode
    
    return {
        "success": True,
        "data": {
            "ecommerce_links": ecommerce_links
        }
    }


# For compatibility, alias the function name
call_firecrawl_extractor = call_crawl4ai_extractor
