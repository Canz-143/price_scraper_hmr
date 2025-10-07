import os
import json
import asyncio
import nest_asyncio
import re
from urllib.parse import urlparse, parse_qs
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, LLMConfig, CacheMode, MemoryAdaptiveDispatcher, RateLimiter
from crawl4ai.extraction_strategy import LLMExtractionStrategy, JsonCssExtractionStrategy
from pydantic import BaseModel, Field

# For compatibility with some environments like Jupyter notebooks
# nest_asyncio.apply()

# --- Configuration ---
# You must set this environment variable for the code to work
# os.environ['GOOGLE_API_KEY'] = 'YOUR_GOOGLE_API_KEY'

# --- Pydantic Model for Data Extraction ---
class ProductPrice(BaseModel):
    combined_price: str = Field(..., description="The product price as a single string, including the currency symbol (e.g., $2000).")
    price: str = Field(..., description="The numerical price as a string, without the currency symbol (e.g., 2000).")
    currency_code: str = Field(..., description="The currency code (e.g., USD, EUR).")
    website_name: str = Field(..., description="The name of the website.")
    product_page_url: str = Field(..., description="The direct URL to the product page.")

# --- URL Filtering Functions (unchanged, but included for completeness) ---
def is_valid_url(url):
    """Checks if a URL is valid and has proper format for Crawl4AI."""
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
    """Determines if a URL is a search or collection page."""
    if not url: return True
    parsed_url = urlparse(url.lower())
    path = parsed_url.path
    query_params = parse_qs(parsed_url.query)
    search_indicators = [r'/search', r'/results', r'/find', r'/query', r'/s/', r'/buscar', r'/recherche', r'/suche']
    collection_indicators = [r'/category', r'/categories', r'/collection', r'/collections', r'/browse', r'/catalog', r'/products(?:/(?:all|list))?$', r'/items', r'/list', r'/archive', r'/tag/', r'/tags/', r'/c/', r'/cat/', r'/department', r'/shop(?:/(?:all|category))?$']
    if any(re.search(pattern, path) for pattern in search_indicators): return True
    if any(re.search(pattern, path) for pattern in collection_indicators): return True
    search_params = ['q', 'query', 'search', 'keyword', 'term', 'find', 's', 'k', 'p']
    if any(param in query_params for param in search_params): return True
    collection_params = ['category', 'cat', 'collection', 'tag', 'filter', 'sort']
    if sum(1 for param in collection_params if param in query_params) >= 2: return True
    pagination_params = ['page', 'p', 'offset', 'start', 'limit']
    if any(param in query_params for param in pagination_params) and sum(1 for param in collection_params if param in query_params) >= 1: return True
    domain = parsed_url.netloc
    if 'amazon.' in domain and ('/s?' in url or '/s/' in path or re.search(r'/b/|/gp/browse/|/departments/', path)): return True
    if 'ebay.' in domain and ('/sch/' in path or '/b/' in path): return True
    if 'shopify' in domain and '/collections/' in path and not re.search(r'/collections/[^/]+/products/', path): return True
    if 'etsy.' in domain and ('/search/' in path or '/c/' in path): return True
    if 'walmart.' in domain and ('/search/' in path or '/browse/' in path): return True
    if 'target.' in domain and ('/s/' in path or '/c/' in path): return True
    return False

def is_likely_product_page(url):
    """Checks for patterns that indicate a URL is a product page."""
    if not url: return False
    parsed_url = urlparse(url.lower())
    path = parsed_url.path
    product_indicators = [r'/product/', r'/item/', r'/p/', r'/dp/', r'/itm/', r'/listing/', r'/products/[^/]+$', r'/[^/]+-p-\d+', r'/\d+\.html?$']
    if any(re.search(pattern, path) for pattern in product_indicators): return True
    path_parts = [part for part in path.split('/') if part]
    if path_parts:
        last_part = path_parts[-1]
        if re.search(r'^[a-zA-Z0-9\-_]+$', last_part) and len(last_part) > 3: return True
    return False

# --- Main Scraper Function ---
async def call_crawl4ai_extractor(links, request_id=None):
    """
    Scrapes product price data from a list of URLs using a hybrid extraction approach.
    It generates a fast schema from the first URL and reuses it for the rest.
    """
    # 1. Filter links to ensure they are valid and likely product pages
    filtered_links = [
        url for url in links[:5]
        if is_valid_url(url) and not is_search_or_collection_page(url) and is_likely_product_page(url)
    ]

    if not filtered_links:
        return {"success": False, "error": "No valid product URLs after filtering."}

    api_token = os.getenv('GOOGLE_API_KEY')
    if not api_token:
        return {"success": False, "error": "GOOGLE_API_KEY environment variable is not set."}

    # 2. Configure a dispatcher for intelligent concurrency and rate-limiting
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=80.0,
        max_session_permit=15,
        rate_limiter=RateLimiter(base_delay=(0.5, 1.5))
    )

    output = []
    
    async with AsyncWebCrawler(verbose=True) as crawler:
        # Step 1: LLM-based Schema Generation (one-time cost)
        llm_config = LLMConfig(provider="gemini/gemini-1.5-flash", api_token=api_token)
        one_time_llm_strategy = LLMExtractionStrategy(
            llm_config=llm_config,
            schema=ProductPrice.model_json_schema(),
            extraction_type="schema",
            instruction="Extract the main product price, currency, website name, and direct URL from this page."
        )

        first_url = filtered_links[0]
        print(f"Starting one-time LLM extraction on {first_url} to generate schema...")
        
        # We process the first URL alone to ensure we have a schema to use for the rest
        results_gen = await crawler.arun_many(
            urls=[first_url],
            config=CrawlerRunConfig(
                cache_mode=CacheMode.ENABLED,
                extraction_strategy=one_time_llm_strategy,
                stream=False # Force wait for the single result
            ),
            dispatcher=dispatcher
        )

        first_result = results_gen[0]
        if not first_result.success or not first_result.extracted_content:
            return {"success": False, "error": f"Failed to get LLM extraction from first URL: {first_result.error_message}"}
        
        try:
            sample_json = json.loads(first_result.extracted_content)
            print("✅ LLM extraction successful. Generating non-LLM schema from HTML.")
            generated_schema = await JsonCssExtractionStrategy.generate_schema(
                html=first_result.cleaned_html,
                target_json_example=json.dumps(sample_json, indent=2),
                llm_config=llm_config
            )
            print("✅ Schema generated. Switching to fast, non-LLM extraction.")
        except (json.JSONDecodeError, Exception) as e:
            return {"success": False, "error": f"Schema generation failed: {e}"}

        # Add the first result to the output list
        output.append({"url": first_result.url, "data": sample_json, "success": True})

        # Step 2: Fast, Schema-based Extraction for Remaining URLs
        schema_strategy = JsonCssExtractionStrategy(schema=generated_schema)
        crawler_config = CrawlerRunConfig(
            cache_mode=CacheMode.ENABLED,
            extraction_strategy=schema_strategy,
            stream=True
        )
        remaining_urls = filtered_links[1:]
        
        if remaining_urls:
            print(f"Crawling remaining {len(remaining_urls)} URLs with the fast schema...")
            async for result in await crawler.arun_many(urls=remaining_urls, config=crawler_config, dispatcher=dispatcher):
                if result.success and result.extracted_content:
                    try:
                        extracted_data = json.loads(result.extracted_content)
                        output.append({"url": result.url, "data": extracted_data, "success": True})
                    except json.JSONDecodeError:
                        output.append({"url": result.url, "error": "Error decoding JSON", "success": False})
                else:
                    output.append({"url": result.url, "error": result.error_message, "success": False})

    return output

# For compatibility, alias the function name
call_firecrawl_extractor = call_crawl4ai_extractor

# --- Example of how to use the function ---
async def main():
    # Replace with your actual list of URLs
    sample_urls = [
        "https://www.example-shop.com/product/xyz123",
        "https://www.another-site.net/products/item-456",
        "https://www.example-shop.com/category/electronics", # This will be filtered out
        "https://www.third-site.co.uk/p/product-789",
    ]
    results = await call_crawl4ai_extractor(sample_urls)
    print("\n--- Final Results ---")
    print(json.dumps(results, indent=2))

# To run the example, uncomment the following line
# if __name__ == "__main__":
#     asyncio.run(main())
