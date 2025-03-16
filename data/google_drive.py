import requests
import re

CHUNK_SIZE = 32768

def download_file_from_google_drive(file_id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)

def get_confirm_token(response):
    # First, try to get the token from cookies.
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    # If not found in cookies, parse the HTML to extract the confirm token.
    match = re.search(r'name="confirm" value="([^"]+)"', response.text)
    if match:
        return match.group(1)
    return None

def save_response_content(response, destination):
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)