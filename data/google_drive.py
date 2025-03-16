import requests
import re

CHUNK_SIZE = 32768
def download_file_from_google_drive(file_id, destination):
    # Base URL for the initial request
    BASE_URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    # Initial request to get the warning page (if any)
    response = session.get(BASE_URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        # Extract the download URL from the HTML form's action attribute
        download_url = get_download_url(response)
        if not download_url:
            download_url = BASE_URL  # Fallback to the base URL if not found

        params = {'id': file_id, 'confirm': token}
        response = session.get(download_url, params=params, stream=True)

    save_response_content(response, destination)

def get_confirm_token(response):
    # Try to retrieve token from cookies first.
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    # Fallback: search for the token in the HTML hidden input field.
    match = re.search(r'name="confirm" value="([^"]+)"', response.text)
    if match:
        return match.group(1)
    return None

def get_download_url(response):
    # Parse the form's action attribute to get the correct URL.
    match = re.search(r'action="(https://[^"]+)"', response.text)
    if match:
        return match.group(1)
    return None

def save_response_content(response, destination):
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # Filter out keep-alive chunks.
                f.write(chunk)