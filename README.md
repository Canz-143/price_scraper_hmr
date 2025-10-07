# Product Finder API

## Overview
A FastAPI-based service for uploading and analyzing product images using Gemini API.

## Setup
1. Clone the repo and navigate to the project directory.
2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   .\venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Add your Google API key to the `.env` file:
   ```
   GOOGLE_API_KEY=your_google_api_key_here
   ```
5. Run the API:
   ```
   uvicorn app.main:app --reload
   ```

## Endpoints
- `POST /analyze-image/` — Upload an image for analysis.

## File Structure
```
app/
  main.py
  routes/
    product.py
  services/
    gemini.py
  utils/
    image_tools.py
  config.py
.env
requirements.txt
README.md
```
