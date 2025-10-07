import os
import json
import asyncio
import re
from urllib.parse import urlparse, parse_qs
from typing import List, Dict, Any

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, LLMConfig, CacheMode
from crawl4ai.extraction_strategy import LLMExtractionStrategy, JsonCssExtractionStrategy
from crawl4ai import MemoryAdaptiveDispatcher, RateLimiter
from pydantic import BaseModel, Field

from app.config import GOOGLE_API_KEY


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
    """Check if URL is valid and has proper format."""
    if not url or not url.startswith(('http://', 'https://')):
        return False
    
    try:
        parsed = urlparse(url)
        if not parsed.netloc or '.' not in parsed.netloc:
            return False
        if '/blocked?' in url or 'blocked' in parsed.path:
            return False
        return True
    except Exception:
        return False


def is_search_or_collection_page(url):
    """Determine if a URL is a search result page or collection page."""
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
    if any(param in query_params for param in search_params):
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
        if '/s?' in url or '/s/' in path or re.search(r'/b/|/gp/browse/|/departments/', path):
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
    """Additional check to identify likely product pages."""
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


def filter_urls(links: List[str], max_urls: int = 10) -> List[str]:
    """Filter and validate URLs in a single pass."""
    filtered = []
    for url in links[:max_urls]:
        if (is_valid_url(url) and 
            not is_search_or_collection_page(url) and 
            is_likely_product_page(url)):
            filtered.append(url)
    return filtered


# OPTIMIZATION 1: Try non-LLM extraction first (1000x faster)
async def try_fast_extraction(urls: List[str], crawler: AsyncWebCrawler) -> Dict[str, Any]:
    """
    Attempt fast CSS-based extraction first before falling back to LLM.
    This is 1000x faster and free compared to LLM extraction.
    """
    # Common CSS selectors for price extraction across major e-commerce sites
    price_schema = {
        "name": "ProductPrice",
        "baseSelector": "body",
        "fields": [
            {
                "name": "combined_price",
                "selector": """
                    [class*='price']:not([class*='strike']):not([class*='was']),
                    [id*='price'],
                    [data-price],
                    .a-price .a-offscreen,
                    span[class*='Price'],
                    meta[property='og:price:amount']
                """,
                "type": "text",
                "default": ""
            },
            {
                "name": "currency_code",
                "selector": "meta[property='og:price:currency']",
                "type": "attribute",
                "attribute": "content",
                "default": "USD"
            }
        ]
    }
    
    fast_strategy = JsonCssExtractionStrategy(price_schema)
    fast_config = CrawlerRunConfig(
        cache_mode=CacheMode.ENABLED,  # Use cache for repeated URLs
        extraction_strategy=fast_strategy,
        word_count_threshold=10,
        page_timeout=15000,  # Faster timeout
        wait_for="css:body",  # Don't wait unnecessarily
    )
    
    results = {}
    
    # arun_many returns a list, not an async iterator
    crawl_results = await crawler.arun_many(urls, config=fast_config)
    
    for result in crawl_results:
        if result.success and result.extracted_content:
            try:
                data = json.loads(result.extracted_content)
                # Check if we got meaningful data
                if data and isinstance(data, list) and len(data) > 0:
                    if data[0].get('combined_price'):
                        results[result.url] = {
                            'method': 'fast',
                            'data': data[0],
                            'success': True
                        }
                        continue
            except:
                pass
        
        # Mark for LLM fallback
        results[result.url] = {'method': 'needs_llm', 'success': False}
    
    return results


# OPTIMIZATION 2: Batch LLM processing with concurrent execution
async def call_crawl4ai_extractor(links: List[str], request_id=None) -> List[Dict[str, Any]]:
    """
    Optimized extraction with multiple strategies:
    1. Fast CSS extraction (try first)
    2. LLM extraction (fallback for failed URLs)
    3. Memory-adaptive concurrency
    4. Concurrent processing for better throughput
    """
    
    # Filter URLs efficiently
    filtered_links = filter_urls(links, max_urls=10)
    
    if not filtered_links:
        return {
            "success": False,
            "error": "No valid product URLs after filtering."
        }
    
    # OPTIMIZATION 3: Configure memory-adaptive dispatcher for better resource management
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=75.0,  # Adjust based on your server
        max_session_permit=8,  # Concurrent crawls (tune based on your server)
        check_interval=0.5,
        rate_limiter=RateLimiter(
            base_delay=(0.1, 0.3),  # Faster delays for better throughput
            max_delay=10.0,
            max_retries=2  # Fewer retries for speed
        )
    )
    
    output = []
    
    async with AsyncWebCrawler(verbose=False) as crawler:  # Disable verbose for speed
        # Step 1: Try fast extraction first
        fast_results = await try_fast_extraction(filtered_links, crawler)
        
        # Collect URLs that need LLM
        llm_needed_urls = [
            url for url, result in fast_results.items() 
            if result.get('method') == 'needs_llm'
        ]
        
        # Add successful fast extractions to output
        for url, result in fast_results.items():
            if result.get('success'):
                parsed = urlparse(url)
                website_name = parsed.netloc.replace('www.', '')
                
                output.append({
                    "url": url,
                    "data": {
                        "combined_price": result['data'].get('combined_price', ''),
                        "price": re.sub(r'[^\d.]', '', result['data'].get('combined_price', '')),
                        "currency_code": result['data'].get('currency_code', 'USD'),
                        "website_name": website_name,
                        "product_page_url": url
                    },
                    "success": True,
                    "method": "fast_css"
                })
        
        # Step 2: Use LLM only for URLs that failed fast extraction
        if llm_needed_urls:
            os.environ['GEMINI_API_KEY'] = GOOGLE_API_KEY
            
            # OPTIMIZATION 4: Use cheaper, faster LLM model
            extraction_strategy = LLMExtractionStrategy(
                llm_config=LLMConfig(
                    provider="gemini/gemini-2.5-flash",  # Already using fast model ✓
                    api_token=os.getenv('GEMINI_API_KEY'),
                    temperature=0.0,  # Deterministic for consistency
                    max_tokens=500  # Limit tokens for faster responses
                ),
                schema=ProductPrice.model_json_schema(),
                extraction_type="schema",
                instruction=(
                    "Extract ONLY: price with currency symbol, numeric price, "
                    "currency code (USD/EUR/etc), website name, and product URL. "
                    "Be concise."
                ),
                input_format="fit_markdown",  # Use cleaned content for speed
                apply_chunking=False,  # No chunking for product pages
            )
            
            llm_config = CrawlerRunConfig(
                cache_mode=CacheMode.ENABLED,  # Cache LLM results
                extraction_strategy=extraction_strategy,
                word_count_threshold=20,
                page_timeout=20000,
                excluded_tags=['nav', 'footer', 'header', 'aside'],  # Skip irrelevant content
            )
            
            # OPTIMIZATION 5: Concurrent processing with dispatcher
            llm_results = await crawler.arun_many(
                urls=llm_needed_urls,
                config=llm_config,
                dispatcher=dispatcher
            )
            
            for result in llm_results:
                if result.success:
                    try:
                        extracted_data = json.loads(result.extracted_content)
                        output.append({
                            "url": result.url,
                            "data": extracted_data,
                            "success": True,
                            "method": "llm"
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
    
    return output


# For compatibility
call_firecrawl_extractor = call_crawl4ai_extractor
