from langchain_community.document_loaders import WebBaseLoader
import requests
from bs4 import BeautifulSoup
import json

NIM_DOCS_URL = "https://docs.nvidia.com/nim/index.html"

# extract all links from the page within class="Chapter-chapters"
def extract_links_from_chapter_chapters(url):
    # Send a GET request to the URL
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for HTTP errors
    
    # Parse the HTML content
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find all elements with class="Chapter-chapters"
    chapter_chapters_divs = soup.find_all(class_="Chapter-chapters")
    
    links = []
    
    # Extract all links from these elements
    for div in chapter_chapters_divs:
        # Find all anchor tags within the div
        anchors = div.find_all('a')
        
        # Extract href and text from each anchor
        for anchor in anchors:
            href = anchor.get('href')
            text = anchor.get_text(strip=True)
            
            # Some links might be relative, convert them to absolute URLs
            if href and not href.startswith(('http://', 'https://')):
                # Handle relative URLs
                if href.startswith('/'):
                    # Absolute path relative to domain
                    base_url = '/'.join(url.split('/')[:3])  # Get domain part (e.g., https://docs.nvidia.com)
                    href = base_url + href
                else:
                    # Relative to current page
                    href = '/'.join(url.split('/')[:-1]) + '/' + href
            
            if href:  # Only add if href is not None or empty
                links.append(href)
    
    return links

# Extract links from the specified class
nim_links = extract_links_from_chapter_chapters(NIM_DOCS_URL)

# Print the results
print(f"Found {len(nim_links)} links within class='Chapter-chapters':")

# Original LangChain document loading code for reference
loader = WebBaseLoader(nim_links)
docs = loader.load()
print("Loaded documents:", len(docs))