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


def parse_price_string(price_text):
    """
    Universal price parser that handles multiple formats and currencies.
    
    Handles:
    - US format: $1,999.99
    - European format: €1.999,99
    - Indian format: ₹1,00,000
    - Various currency symbols and codes
    
    Returns:
        tuple: (numeric_price_string, currency_code) or (None, None) if parsing fails
    """
    if not price_text or not isinstance(price_text, str):
        return None, None
    
    # Remove HTML entities and extra whitespace
    price_text = re.sub(r'&[a-z]+;', '', price_text, flags=re.IGNORECASE)
    price_text = ' '.join(price_text.split())
    
    # Comprehensive currency mapping
    currency_map = {
        # Symbols
        '$': 'USD',
        'US$': 'USD',
        'USD': 'USD',
        '₱': 'PHP',
        'PHP': 'PHP',
        '€': 'EUR',
        'EUR': 'EUR',
        '£': 'GBP',
        'GBP': 'GBP',
        '¥': 'JPY',
        'JPY': 'JPY',
        'CNY': 'CNY',
        '₹': 'INR',
        'Rs': 'INR',
        'INR': 'INR',
        'R$': 'BRL',
        'BRL': 'BRL',
        'CHF': 'CHF',
        'CAD': 'CAD',
        'C$': 'CAD',
        'AUD': 'AUD',
        'A$': 'AUD',
        'NZD': 'NZD',
        'SGD': 'SGD',
        'S$': 'SGD',
        'HKD': 'HKD',
        'HK$': 'HKD',
        'MXN': 'MXN',
        'ZAR': 'ZAR',
        'R': 'ZAR',
        'KRW': 'KRW',
        '₩': 'KRW',
        'THB': 'THB',
        '฿': 'THB',
        'MYR': 'MYR',
        'RM': 'MYR',
        'IDR': 'IDR',
        'Rp': 'IDR',
        'VND': 'VND',
        '₫': 'VND',
    }
    
    # Detect currency code
    currency_code = None
    for symbol, code in currency_map.items():
        if symbol in price_text:
            currency_code = code
            # Remove the currency symbol/code for numeric extraction
            price_text = price_text.replace(symbol, ' ')
            break
    
    # Remove common price-related words
    price_text = re.sub(r'\b(price|sale|was|now|from|starting|only)\b', '', price_text, flags=re.IGNORECASE)
    
    # Extract only numeric characters, commas, periods, and spaces
    numeric_text = re.sub(r'[^\d,\.\s]', '', price_text)
    numeric_text = numeric_text.strip()
    
    if not numeric_text:
        return None, currency_code
    
    # Handle different number formats
    # Strategy: Determine format by analyzing separators
    
    # Count occurrences
    comma_count = numeric_text.count(',')
    period_count = numeric_text.count('.')
    space_count = numeric_text.count(' ')
    
    # Remove spaces (used as thousands separators in some locales)
    numeric_text = numeric_text.replace(' ', '')
    
    # Determine the decimal separator
    if comma_count > 0 and period_count > 0:
        # Both present - last one is decimal separator
        last_comma_pos = numeric_text.rindex(',')
        last_period_pos = numeric_text.rindex('.')
        
        if last_comma_pos > last_period_pos:
            # European format: 1.999,99
            numeric_text = numeric_text.replace('.', '').replace(',', '.')
        else:
            # US format: 1,999.99
            numeric_text = numeric_text.replace(',', '')
            
    elif comma_count > 0:
        # Only commas present
        parts = numeric_text.split(',')
        
        # If last part has 2 digits, it's likely a decimal separator
        if len(parts) > 1 and len(parts[-1]) == 2:
            # Format like: 99,99 or 1.999,99
            numeric_text = numeric_text.replace(',', '.')
        else:
            # Thousands separator: 1,999 or 1,999,999
            numeric_text = numeric_text.replace(',', '')
            
    elif period_count > 1:
        # Multiple periods - thousands separator (European)
        # Format: 1.999.999
        numeric_text = numeric_text.replace('.', '')
        
    # If there's still a period, assume it's decimal
    # If no period, it's a whole number
    
    # Final validation and conversion
    try:
        price_value = float(numeric_text)
        
        # Sanity check: price should be positive and reasonable
        if price_value <= 0:
            return None, currency_code
        if price_value > 10000000:  # 10 million - adjust as needed
            return None, currency_code
        
        # Return as string to preserve decimal precision
        return str(price_value), currency_code
        
    except ValueError:
        return None, currency_code


def validate_and_fix_extracted_price(extracted_data):
    """
    Validate and fix extracted price data using the robust parser.
    This acts as a post-processing layer after LLM extraction.
    
    Args:
        extracted_data (dict): Data extracted by LLM
    
    Returns:
        dict: Validated and fixed price data
    """
    price = extracted_data.get("price", "")
    combined = extracted_data.get("combined_price", "")
    currency = extracted_data.get("currency_code", "")
    
    # Flag to track if we made fixes
    was_fixed = False
    
    # Case 1: LLM extracted combined_price but price is empty or invalid
    if combined and (not price or not is_valid_numeric_price(price)):
        parsed_price, parsed_currency = parse_price_string(combined)
        
        if parsed_price:
            extracted_data["price"] = parsed_price
            was_fixed = True
            
            # Also fix currency if it was missing or wrong
            if not currency and parsed_currency:
                extracted_data["currency_code"] = parsed_currency
                was_fixed = True
    
    # Case 2: Price exists but might have formatting issues
    elif price:
        # Try to clean/validate the price
        cleaned_price, detected_currency = parse_price_string(price)
        
        if cleaned_price and cleaned_price != price:
            extracted_data["price"] = cleaned_price
            was_fixed = True
        elif not cleaned_price:
            # Price is invalid, try combined_price
            if combined:
                parsed_price, parsed_currency = parse_price_string(combined)
                if parsed_price:
                    extracted_data["price"] = parsed_price
                    was_fixed = True
    
    # Case 3: Validate currency code
    if currency:
        # Ensure it's uppercase and valid
        currency_upper = currency.upper()
        valid_currencies = [
            'USD', 'EUR', 'GBP', 'JPY', 'CNY', 'INR', 'PHP', 'BRL',
            'CAD', 'AUD', 'NZD', 'SGD', 'HKD', 'MXN', 'ZAR', 'KRW',
            'THB', 'MYR', 'IDR', 'VND', 'CHF'
        ]
        
        if currency_upper in valid_currencies:
            extracted_data["currency_code"] = currency_upper
        else:
            # Try to detect from combined_price
            if combined:
                _, detected_currency = parse_price_string(combined)
                if detected_currency:
                    extracted_data["currency_code"] = detected_currency
                    was_fixed = True
    
    # Case 4: Currency is missing but we have a price
    if not currency and (price or combined):
        _, detected_currency = parse_price_string(combined or price)
        if detected_currency:
            extracted_data["currency_code"] = detected_currency
            was_fixed = True
    
    # Add validation metadata
    extracted_data["_validation_passed"] = is_valid_numeric_price(extracted_data.get("price"))
    extracted_data["_was_auto_fixed"] = was_fixed
    
    return extracted_data


def is_valid_numeric_price(price_str):
    """Check if a price string is a valid numeric value."""
    if not price_str:
        return False
    try:
        price_float = float(price_str)
        return 0 < price_float < 10000000
    except (ValueError, TypeError):
        return False


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
    OPTIMIZED: Extract product information using Crawl4AI with enhanced price parsing.
    
    Key improvements:
    - Universal price parser handles all formats and currencies
    - Automatic validation and fixing of LLM extraction errors
    - Post-processing layer ensures price accuracy
    
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
    
    # Enhanced LLM extraction with clearer price examples
    extraction_strategy = LLMExtractionStrategy(
        llm_config=LLMConfig(
            provider="gemini/gemini-2.0-flash",
            api_token=api_token
        ),
        schema=ProductPrice.model_json_schema(),
        extraction_type="schema",
        instruction=(
            "You are a precise product price extractor. Extract the MAIN/FEATURED product's information.\n\n"
            
            "🎯 CRITICAL: Extract prices EXACTLY as they appear on the page.\n\n"
            
            "📋 FIELD INSTRUCTIONS:\n"
            "1. combined_price: Copy the COMPLETE price text as displayed\n"
            "   Examples:\n"
            "   - '$2,499.99'\n"
            "   - '₱2,500.00'\n"
            "   - '€1.999,99'\n"
            "   - 'US $49.99'\n"
            "   - '£1,234.56'\n"
            "   - '¥10,000'\n\n"
            
            "2. price: Extract ONLY the raw number (critical for comparison)\n"
            "   Rules:\n"
            "   - Keep ONLY digits and ONE decimal point\n"
            "   - Remove ALL symbols: $, €, ₱, £, ¥, Rs, etc.\n"
            "   - Remove thousand separators (commas, spaces, periods used as thousands)\n"
            "   Examples:\n"
            "   - '$2,499.99' → '2499.99'\n"
            "   - '₱2,500.00' → '2500.00'\n"
            "   - '€1.999,99' → '1999.99' (note: European comma becomes period)\n"
            "   - 'Rs 1,00,000' → '100000'\n\n"
            
            "3. currency_code: 3-letter ISO code\n"
            "   Common codes: USD, PHP, EUR, GBP, JPY, INR, CAD, AUD, SGD, HKD, MXN\n\n"
            
            "4. website_name: E-commerce site name (e.g., 'Amazon', 'eBay', 'Shopify Store')\n\n"
            
            "5. product_page_url: The exact URL of this product page\n\n"
            
            "💰 PRICE SELECTION RULES:\n"
            "✓ Choose the CURRENT/ACTIVE price (usually near 'Add to Cart' button)\n"
            "✓ For 'from $X' or 'starting at $X', extract X\n"
            "✓ For price ranges '$100-$200', extract the LOWER price: '100'\n\n"
            
            "✗ IGNORE these:\n"
            "- Shipping costs\n"
            "- Tax amounts\n"
            "- Monthly payment plans\n"
            
            "⚠️ If no clear price is visible, return empty strings for all fields.\n"
            "⚠️ Don't extract shipping/tax as the product price.\n"
        ),
        extra_args={
            "temperature": 0,
            "max_tokens": 15000
        }
    )
    
    # Browser and run configuration
    browser_config = BrowserConfig(
        headless=True,
        java_script_enabled=True,
        accept_downloads=False,
    )
    
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        extraction_strategy=extraction_strategy,
        wait_until="domcontentloaded",
        page_timeout=30000,
        word_count_threshold=10,
        stream=True,
        semaphore_count=3
    )
    
    ecommerce_links = []
    failed_extractions = []
    
    # Process URLs with streaming
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
                    
                    # Ensure it's a dict
                    if isinstance(extracted_data, dict):
                        # ⭐ KEY IMPROVEMENT: Validate and fix prices using robust parser
                        extracted_data = validate_and_fix_extracted_price(extracted_data)
                        
                        # Only add if we have a valid price
                        if extracted_data.get("_validation_passed"):
                            ecommerce_links.append({
                                "website_url": extracted_data.get("product_page_url", result.url),
                                "price_string": extracted_data.get("price", ""),
                                "website_name": extracted_data.get("website_name", ""),
                                "currency_code": extracted_data.get("currency_code", ""),
                                "price_combined": extracted_data.get("combined_price", ""),
                                "auto_fixed": extracted_data.get("_was_auto_fixed", False)
                            })
                        else:
                            failed_extractions.append({
                                "url": result.url,
                                "reason": "Invalid price after parsing"
                            })
                    
                except json.JSONDecodeError as e:
                    failed_extractions.append({
                        "url": result.url,
                        "reason": f"JSON decode error: {str(e)}"
                    })
            else:
                failed_extractions.append({
                    "url": result.url,
                    "reason": result.error_message
                })
    
    # Return results with metadata
    return {
        "success": True,
        "data": {
            "ecommerce_links": ecommerce_links,
            "total_attempted": len(filtered_links),
            "successful_extractions": len(ecommerce_links),
            "failed_extractions": len(failed_extractions),
            "failures": failed_extractions if failed_extractions else None
        }
    }


# For compatibility
call_firecrawl_extractor = call_crawl4ai_extractor
