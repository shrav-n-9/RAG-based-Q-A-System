import os
import requests
from bs4 import BeautifulSoup
import time  # For rate limiting

DATA_DIR = "data/raw"

def download_file(url: str, save_dir: str = DATA_DIR) -> str:
    """Downloads a PDF file from the given URL and saves it locally.
    Returns the saved file path."""
    os.makedirs(save_dir, exist_ok=True)
    filename = url.split("/")[-1].split("?")[0]  # Handle query params
    if not filename.endswith(".pdf"):
        filename += ".pdf"  # Ensure PDF extension
    file_path = os.path.join(save_dir, filename)

    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(file_path, "wb") as f:
        f.write(response.content)

    print(f"Downloaded: {file_path}")
    return file_path

def scrape_links(base_url: str, file_ext: str = ".pdf") -> list:
    """Scrapes transcript PDF links from the Lecture Videos section of the course page."""
    # Use the video galleries URL for lecture videos list
    lecture_videos_url = base_url.rstrip("/") + "/video_galleries/lecture-videos/"
    
    try:
        response = requests.get(lecture_videos_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        links = []
        # Find links to individual lecture resource pages (e.g., /resources/lecture-1-...)
        for a in soup.find_all("a", href=True):
            href = a["href"]
            # Flexible match for lecture resource pages
            if "/resources/lecture-" in href and href.count("/") > 2:  # Avoid malformed URLs
                if not href.startswith("http"):
                    full_href = "https://ocw.mit.edu" + href if href.startswith("/") else base_url.rstrip("/") + "/" + href
                else:
                    full_href = href
                print(f"Visiting lecture page: {full_href}")  # Debug
                
                # Rate limiting to avoid redirects/issues
                time.sleep(1)
                
                # Visit each lecture page to find transcript PDF link
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
                lecture_response = requests.get(full_href, headers=headers)
                lecture_response.raise_for_status()
                lecture_soup = BeautifulSoup(lecture_response.text, "html.parser")
                
                # Primary: Look for "Download transcript" or similar link text
                transcript_found = False
                for lecture_a in lecture_soup.find_all("a", href=True):
                    link_text = lecture_a.text.strip().lower()
                    if "download transcript" in link_text or "transcript" in link_text:
                        transcript_href = lecture_a["href"]
                        if not transcript_href.startswith("http"):
                            transcript_href = "https://ocw.mit.edu" + transcript_href if transcript_href.startswith("/") else full_href.rstrip("/") + "/" + transcript_href
                        print(f"Found potential transcript link: {transcript_href} (text: {lecture_a.text.strip()})")  # Debug
                        if transcript_href.endswith(".pdf"):
                            links.append(transcript_href)
                            transcript_found = True
                        else:
                            print(f"Transcript link found but not PDF: {transcript_href}")  # Debug

                # Fallback: Search for any .pdf links on the page, especially those with "transcript" in the URL or text
                if not transcript_found:
                    print(f"No exact 'Download transcript' found on {full_href}, checking for any PDFs...")  # Debug
                    for lecture_a in lecture_soup.find_all("a", href=True):
                        pdf_href = lecture_a["href"]
                        if pdf_href.endswith(file_ext) or "transcript" in pdf_href.lower():
                            if not pdf_href.startswith("http"):
                                pdf_href = "https://ocw.mit.edu" + pdf_href if pdf_href.startswith("/") else full_href.rstrip("/") + "/" + pdf_href
                            print(f"Found fallback PDF link: {pdf_href} (text: {lecture_a.text.strip()})")  # Debug
                            links.append(pdf_href)

        print(f"Found {len(links)} {file_ext} links")
        return links

    except requests.RequestException as e:
        print(f"Error accessing pages: {e}")
        return []

if __name__ == "__main__":
    # MIT OCW Computer Science & Programming in Python
    base_url = "https://ocw.mit.edu/courses/6-0001-introduction-to-computer-science-and-programming-in-python-fall-2016/"

    # Scrape transcript PDF links
    pdf_links = scrape_links(base_url, ".pdf")

    # Download them
    for link in pdf_links:
        download_file(link)