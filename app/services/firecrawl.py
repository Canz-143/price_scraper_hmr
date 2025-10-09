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
from crawl4ai.extraction_strategy import LLMExtractionStrategy, JsonCssExtractionStrategy
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
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


async def call_crawl4ai_extractor(links, request_id=None):
    """
    OPTIMIZED: Extract product information using Crawl4AI with enhanced accuracy and speed.
    
    Strategy: CSS-first extraction with LLM fallback for maximum speed and accuracy
    
    Key optimizations:
    1. CSS extraction first (10x faster, no API cost) - tries common price patterns
    2. LLM fallback only if CSS fails - for complex/dynamic layouts
    3. Content filtering to focus LLM on relevant content
    4. Site-specific JavaScript execution for dynamic prices
    5. Better wait strategies for price rendering
    6. Parallel processing with optimal concurrency
    
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
    
    # OPTIMIZATION 1: CSS-based extraction schema (fast, no API cost)
    # Covers most common e-commerce price patterns
    css_schema = {
        "name": "Product Price Data",
        "baseSelector": "body",
        "fields": [
            # Price - try multiple common selectors
            {
                "name": "price_raw",
                "selector": "[itemprop='price'], [data-price], [class*='price']:not([class*='original']):not([class*='was']):not([class*='strike']), [id*='price'], .sale-price, .current-price, .product-price, #product-price, .price-now, meta[property='og:price:amount']",
                "type": "text",
                "attribute": "content"
            },
            # Currency code
            {
                "name": "currency_code",
                "selector": "[itemprop='priceCurrency'], [data-currency], meta[property='og:price:currency']",
                "type": "text",
                "attribute": "content"
            },
            # Product name/title
            {
                "name": "product_name",
                "selector": "[itemprop='name'], h1.product-title, h1[class*='product'], .product-name, #product-name",
                "type": "text"
            },
            # Website/brand name
            {
                "name": "website_name",
                "selector": "[itemprop='brand'], meta[property='og:site_name'], .site-logo img, header img[alt]",
                "type": "text",
                "attribute": "alt"
            }
        ]
    }
    
    # OPTIMIZATION 2: LLM extraction as fallback (only if CSS fails)
    llm_extraction_strategy = LLMExtractionStrategy(
        llm_config=LLMConfig(
            provider="gemini/gemini-2.0-flash-exp",  # Fastest Gemini model
            api_token=api_token
        ),
        schema=ProductPrice.model_json_schema(),
        extraction_type="schema",
        instruction="""Extract the main product's pricing information with EXTREME PRECISION.

CRITICAL EXTRACTION RULES:

1. COMBINED_PRICE - Extract EXACTLY as displayed:
   ✓ Correct: "$2,499.99", "€1.999,99", "₱2,500.00", "£49.99"
   ✗ Wrong: "2499.99", "USD 2499.99"

2. PRICE - Numeric value ONLY (for comparison):
   Steps:
   a) Remove ALL currency symbols: $, €, ₱, £, ¥, ₹, kr, etc.
   b) Remove thousand separators: commas, spaces, periods used as separators
   c) Keep ONLY decimal point (convert commas to periods if needed)
   
   Examples:
   "$2,499.99" → "2499.99"
   "€1.999,99" → "1999.99" (European format with comma decimal)
   "₱2,500" → "2500"
   "₹1,00,000.00" → "100000.00" (Indian lakh format)
   "49,99 kr" → "49.99"

3. CURRENCY_CODE - ISO 4217 codes:
   USD, EUR, GBP, JPY, CNY, INR, PHP, AUD, CAD, SGD, THB, MYR, IDR, VND, KRW, etc.

4. WEBSITE_NAME - Brand/store name:
   "Amazon", "eBay", "Lazada", "Shopee", "Walmart", "Target", etc.

5. PRODUCT_PAGE_URL - Full canonical URL

PRICE LOCATION HINTS (check in order):
1. Near "Add to Cart" / "Buy Now" buttons
2. Product title area or hero section
3. Price block with class/id containing: price, cost, amount, valor
4. Structured data (JSON-LD) with @type Product
5. Meta tags (og:price:amount, product:price:amount)

PRICE SELECTION PRIORITY:
✓ Current/Sale price (largest, most prominent)
✓ "Now $X" or "Sale $X"
✗ Crossed-out prices (original/was price)
✗ MSRP or "List price"
✗ "From $X" or "Starting at $X" - extract X only
✗ Price ranges "$100-$200" - extract lower: "100"

VALIDATION CHECKS:
- Price must be positive number
- Price should be reasonable (not 0.00, not millions unless luxury)
- Currency must match website's region
- If no clear price visible, return empty strings ""

Return ONE product only (the main/featured item).""",
        extra_args={
            "temperature": 0,  # Deterministic
            "top_p": 0.9,
            "max_tokens": 2000  # Reduced for faster response
        }
    )
    
    # OPTIMIZATION 3: Content filtering to focus on relevant content (reduces LLM processing)
    markdown_generator = DefaultMarkdownGenerator(
        content_filter=PruningContentFilter(
            threshold=0.48,  # Keep more content than default
            threshold_type="fixed"
        )
    )
    
    # OPTIMIZATION 4: Site-specific JavaScript for price extraction
    # This helps ensure dynamic prices are loaded
    price_wait_js = """
    (async () => {
        // Wait for common price selectors to appear
        const priceSelectors = [
            '[class*="price"]', '[id*="price"]',
            '[class*="Price"]', '[id*="Price"]',
            '[data-price]', '[itemprop="price"]',
            '.product-price', '#product-price',
            '.current-price', '.sale-price'
        ];
        
        const maxWait = 3000; // 3 seconds max
        const startTime = Date.now();
        
        while (Date.now() - startTime < maxWait) {
            for (const selector of priceSelectors) {
                const el = document.querySelector(selector);
                if (el && el.textContent.trim()) {
                    return; // Price found
                }
            }
            await new Promise(r => setTimeout(r, 200));
        }
    })();
    """
    
    # OPTIMIZATION 5: Browser config optimized for speed and compatibility
    browser_config = BrowserConfig(
        headless=True,
        java_script_enabled=True,
        accept_downloads=False,
        viewport_width=1920,  # Desktop view often shows prices better
        viewport_height=1080,
        # User agent helps avoid bot detection
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    
    # Helper function to parse price from CSS extraction
    def parse_css_price(css_data, url):
        """Parse and validate price data from CSS extraction."""
        try:
            if not css_data or not isinstance(css_data, dict):
                return None
            
            price_raw = css_data.get('price_raw', '').strip()
            if not price_raw:
                return None
            
            # Extract numeric price
            # Remove currency symbols and format
            price_clean = re.sub(r'[^\d.,]', '', price_raw)
            
            # Handle different decimal separators
            # European format: 1.999,99 -> 1999.99
            if ',' in price_clean and '.' in price_clean:
                if price_clean.rindex(',') > price_clean.rindex('.'):
                    # Comma is decimal separator
                    price_clean = price_clean.replace('.', '').replace(',', '.')
                else:
                    # Period is decimal separator
                    price_clean = price_clean.replace(',', '')
            elif ',' in price_clean:
                # Check if comma is decimal separator (less than 3 digits after)
                parts = price_clean.split(',')
                if len(parts[-1]) <= 2:
                    price_clean = price_clean.replace(',', '.')
                else:
                    price_clean = price_clean.replace(',', '')
            
            # Validate numeric
            try:
                price_float = float(price_clean)
                if price_float <= 0 or price_float > 1000000000:  # Sanity check
                    return None
            except ValueError:
                return None
            
            # Determine currency from various sources
            currency = css_data.get('currency_code', '').strip().upper()
            if not currency:
                # Try to extract from price string
                currency_symbols = {
                    '
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        extraction_strategy=extraction_strategy,
        markdown_generator=markdown_generator,
        
        # Wait for prices to load
        js_code=[price_wait_js],
        wait_for="css:.price, [data-price], [itemprop='price']",  # Wait for price elements
        wait_until="domcontentloaded",  # Faster than networkidle
        page_timeout=35000,  # 35s timeout (slightly longer for price loading)
        
        word_count_threshold=50,  # Skip empty pages
        stream=True,  # Process as results arrive
        semaphore_count=5  # Increased concurrency (adjust based on API limits)
    )
    
    ecommerce_links = []
    failed_count = 0
    
    # OPTIMIZATION 6: Process with streaming for faster results
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
                    
                    if isinstance(extracted_data, dict):
                        # Validate price data
                        price_str = extracted_data.get("price", "").strip()
                        combined_price = extracted_data.get("combined_price", "").strip()
                        
                        # Only add if we have valid price data
                        if price_str and combined_price:
                            ecommerce_links.append({
                                "website_url": extracted_data.get("product_page_url", result.url),
                                "price_string": price_str,
                                "website_name": extracted_data.get("website_name", ""),
                                "currency_code": extracted_data.get("currency_code", ""),
                                "price_combined": combined_price
                            })
                        else:
                            failed_count += 1
                            if request_id:
                                print(f"[{request_id}] No price found for {result.url}")
                    
                except json.JSONDecodeError as e:
                    failed_count += 1
                    if request_id:
                        print(f"[{request_id}] JSON decode error for {result.url}: {e}")
            else:
                failed_count += 1
                if request_id:
                    print(f"[{request_id}] Crawl failed for {result.url}: {result.error_message}")
    
    success_rate = len(ecommerce_links) / len(filtered_links) if filtered_links else 0
    
    return {
        "success": True,
        "data": {
            "ecommerce_links": ecommerce_links
        },
        "metadata": {
            "total_urls": len(filtered_links),
            "successful_extractions": len(ecommerce_links),
            "failed_extractions": failed_count,
            "success_rate": f"{success_rate:.1%}"
        }
    }


# For compatibility
call_firecrawl_extractor = call_crawl4ai_extractor: 'USD', '€': 'EUR', '£': 'GBP', '¥': 'JPY',
                    '₹': 'INR', '₱': 'PHP', 'R
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        extraction_strategy=extraction_strategy,
        markdown_generator=markdown_generator,
        
        # Wait for prices to load
        js_code=[price_wait_js],
        wait_for="css:.price, [data-price], [itemprop='price']",  # Wait for price elements
        wait_until="domcontentloaded",  # Faster than networkidle
        page_timeout=35000,  # 35s timeout (slightly longer for price loading)
        
        word_count_threshold=50,  # Skip empty pages
        stream=True,  # Process as results arrive
        semaphore_count=5  # Increased concurrency (adjust based on API limits)
    )
    
    ecommerce_links = []
    failed_count = 0
    
    # OPTIMIZATION 6: Process with streaming for faster results
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
                    
                    if isinstance(extracted_data, dict):
                        # Validate price data
                        price_str = extracted_data.get("price", "").strip()
                        combined_price = extracted_data.get("combined_price", "").strip()
                        
                        # Only add if we have valid price data
                        if price_str and combined_price:
                            ecommerce_links.append({
                                "website_url": extracted_data.get("product_page_url", result.url),
                                "price_string": price_str,
                                "website_name": extracted_data.get("website_name", ""),
                                "currency_code": extracted_data.get("currency_code", ""),
                                "price_combined": combined_price
                            })
                        else:
                            failed_count += 1
                            if request_id:
                                print(f"[{request_id}] No price found for {result.url}")
                    
                except json.JSONDecodeError as e:
                    failed_count += 1
                    if request_id:
                        print(f"[{request_id}] JSON decode error for {result.url}: {e}")
            else:
                failed_count += 1
                if request_id:
                    print(f"[{request_id}] Crawl failed for {result.url}: {result.error_message}")
    
    success_rate = len(ecommerce_links) / len(filtered_links) if filtered_links else 0
    
    return {
        "success": True,
        "data": {
            "ecommerce_links": ecommerce_links
        },
        "metadata": {
            "total_urls": len(filtered_links),
            "successful_extractions": len(ecommerce_links),
            "failed_extractions": failed_count,
            "success_rate": f"{success_rate:.1%}"
        }
    }


# For compatibility
call_firecrawl_extractor = call_crawl4ai_extractor: 'BRL', 'kr': 'SEK',
                    'zł': 'PLN', '₪': 'ILS', '₩': 'KRW', '฿': 'THB',
                    'RM': 'MYR', 'Rp': 'IDR', '₫': 'VND'
                }
                for symbol, code in currency_symbols.items():
                    if symbol in price_raw:
                        currency = code
                        break
            
            # Get website name from domain or meta
            website_name = css_data.get('website_name', '')
            if not website_name:
                domain = urlparse(url).netloc
                website_name = domain.split('.')[0].title()
            
            return {
                "website_url": url,
                "price_string": price_clean,
                "website_name": website_name,
                "currency_code": currency or 'USD',  # Default to USD
                "price_combined": price_raw
            }
            
        except Exception as e:
            if request_id:
                print(f"[{request_id}] CSS price parsing error: {e}")
            return None
    
    # OPTIMIZATION 6: Run config - Try CSS first
    css_run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        extraction_strategy=JsonCssExtractionStrategy(css_schema),
        
        # Wait for prices to load
        js_code=[price_wait_js],
        wait_for="css:[class*='price' i], [data-price], [itemprop='price']",
        wait_until="domcontentloaded",
        page_timeout=35000,
        
        word_count_threshold=50,
        stream=True,
        semaphore_count=5
    )
    
    # OPTIMIZATION 7: LLM fallback config (only for failed CSS extractions)
    llm_run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        extraction_strategy=llm_extraction_strategy,
        markdown_generator=markdown_generator,
        
        js_code=[price_wait_js],
        wait_for="css:[class*='price' i], [data-price], [itemprop='price']",
        wait_until="domcontentloaded",
        page_timeout=35000,
        
        word_count_threshold=50,
        stream=False,  # No streaming for fallback
        semaphore_count=3  # Lower concurrency for LLM
    )
    
    ecommerce_links = []
    failed_css_urls = []
    css_success = 0
    llm_success = 0
    failed_count = 0
    
    # PHASE 1: Try CSS extraction first (fast, no cost)
    if request_id:
        print(f"[{request_id}] Phase 1: Attempting CSS extraction for {len(filtered_links)} URLs...")
    
    async with AsyncWebCrawler(config=browser_config, verbose=True) as crawler:
        async for result in await crawler.arun_many(urls=filtered_links, config=css_run_config):
            if result.success:
                try:
                    css_data = json.loads(result.extracted_content)
                    
                    # Handle list response
                    if isinstance(css_data, list):
                        css_data = css_data[0] if css_data else {}
                    
                    # Try to parse CSS extraction
                    parsed_result = parse_css_price(css_data, result.url)
                    
                    if parsed_result and parsed_result.get('price_string'):
                        ecommerce_links.append(parsed_result)
                        css_success += 1
                        if request_id:
                            print(f"[{request_id}] ✓ CSS extracted: {result.url}")
                    else:
                        # CSS failed, mark for LLM fallback
                        failed_css_urls.append(result.url)
                        if request_id:
                            print(f"[{request_id}] ⚠ CSS failed, queuing for LLM: {result.url}")
                
                except (json.JSONDecodeError, Exception) as e:
                    failed_css_urls.append(result.url)
                    if request_id:
                        print(f"[{request_id}] ⚠ CSS error, queuing for LLM: {result.url}")
            else:
                failed_count += 1
                if request_id:
                    print(f"[{request_id}] ✗ Crawl failed: {result.url}")
    
    # PHASE 2: LLM fallback for failed CSS extractions
    if failed_css_urls:
        if request_id:
            print(f"[{request_id}] Phase 2: LLM fallback for {len(failed_css_urls)} URLs...")
        
        async with AsyncWebCrawler(config=browser_config, verbose=True) as crawler:
            results = await crawler.arun_many(urls=failed_css_urls, config=llm_run_config)
            
            for result in results:
                if result.success:
                    try:
                        extracted_data = json.loads(result.extracted_content)
                        
                        if isinstance(extracted_data, list):
                            extracted_data = extracted_data[0] if extracted_data else {}
                        
                        if isinstance(extracted_data, dict):
                            price_str = extracted_data.get("price", "").strip()
                            combined_price = extracted_data.get("combined_price", "").strip()
                            
                            if price_str and combined_price:
                                ecommerce_links.append({
                                    "website_url": extracted_data.get("product_page_url", result.url),
                                    "price_string": price_str,
                                    "website_name": extracted_data.get("website_name", ""),
                                    "currency_code": extracted_data.get("currency_code", ""),
                                    "price_combined": combined_price
                                })
                                llm_success += 1
                                if request_id:
                                    print(f"[{request_id}] ✓ LLM extracted: {result.url}")
                            else:
                                failed_count += 1
                                if request_id:
                                    print(f"[{request_id}] ✗ LLM no price: {result.url}")
                    
                    except (json.JSONDecodeError, Exception) as e:
                        failed_count += 1
                        if request_id:
                            print(f"[{request_id}] ✗ LLM error: {result.url}")
                else:
                    failed_count += 1
    
    total_success = css_success + llm_success
    success_rate = total_success / len(filtered_links) if filtered_links else 0
    
    if request_id:
        print(f"[{request_id}] Results: {css_success} CSS, {llm_success} LLM, {failed_count} failed")
    
    return {
        "success": True,
        "data": {
            "ecommerce_links": ecommerce_links
        },
        "metadata": {
            "total_urls": len(filtered_links),
            "css_extractions": css_success,
            "llm_extractions": llm_success,
            "failed_extractions": failed_count,
            "success_rate": f"{success_rate:.1%}",
            "cost_savings": f"{css_success}/{total_success} extractions were free (CSS)"
        }
    }


# For compatibility
call_firecrawl_extractor = call_crawl4ai_extractor
