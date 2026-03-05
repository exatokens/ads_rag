"""
gmail_extractor.py

Extracts promotional emails from Gmail including:
- Plain text + HTML text
- Inline and attachment images
- External images from <img src="...">
- Hidden preheader text
- Alt texts

Improvement over original:
- Returns image paths in a clean list ready for Dimension 1 CLIP pipeline
- Tracks image source type (inline / attachment / external)
- Skips tracking pixels (images smaller than 10x10 or 1x1 in URL pattern)
- Adds image_index to each image for ordering
"""

import os
import base64
import json
import re
import requests
from urllib.parse import urlparse

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from bs4 import BeautifulSoup

# Scope — read only, promotions tab filtered at query time
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']


# ---------------------------------------------------------------------------
# AUTH
# ---------------------------------------------------------------------------

def get_gmail_service():
    """
    Authenticate and return Gmail API service.
    Caches token in token.json for subsequent runs.
    """
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials3.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    return build('gmail', 'v1', credentials=creds)


# ---------------------------------------------------------------------------
# TRACKING PIXEL DETECTION
# ---------------------------------------------------------------------------

def is_tracking_pixel(src: str, alt: str) -> bool:
    """
    Detect tracking pixel images that carry no deal information.
    These are 1x1 or tiny images used only for open-rate tracking.
    """
    if not src:
        return False

    # Common tracking pixel URL patterns
    tracking_patterns = [
        r'track', r'pixel', r'beacon', r'open', r'click',
        r'1x1', r'spacer', r'clear\.gif', r'blank\.gif'
    ]
    src_lower = src.lower()
    for pattern in tracking_patterns:
        if re.search(pattern, src_lower):
            return True

    # Images with no alt text and suspicious URL structure
    # (very short URLs with random tokens = tracking)
    parsed = urlparse(src)
    path = parsed.path
    if len(path) < 5 and not alt:
        return True

    return False


# ---------------------------------------------------------------------------
# CORE EXTRACTION
# ---------------------------------------------------------------------------

def extract_email_with_images(service, msg_id: str, base_dir: str = "downloaded_emails") -> dict:
    """
    Extract email content including text and all images.

    Returns structured dict ready for Dimension 1 CLIP pipeline:
    {
        'id':            email message ID,
        'subject':       email subject,
        'sender':        sender address,
        'date':          send date,
        'text_plain':    raw plain text,
        'text_html':     raw HTML,
        'text_cleaned':  assembled searchable text corpus,
        'preheader':     hidden preview text (often best deal summary),
        'alt_texts':     list of alt text strings from images,
        'images':        list of image dicts (see below),
        'folder_path':   local folder where files are saved
    }

    Each image dict:
    {
        'image_index':  int (order found in email),
        'filename':     str,
        'path':         str (local file path — ready for CLIP),
        'source_url':   str or None (original URL if external),
        'alt_text':     str,
        'source_type':  'inline' | 'attachment' | 'external',
        'mime_type':    str
    }
    """
    os.makedirs(base_dir, exist_ok=True)

    # Fetch full message
    msg = service.users().messages().get(userId='me', id=msg_id, format='full').execute()
    payload = msg.get('payload', {})
    headers = payload.get('headers', [])

    # Extract metadata
    subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
    sender  = next((h['value'] for h in headers if h['name'] == 'From'), 'Unknown')
    date    = next((h['value'] for h in headers if h['name'] == 'Date'), 'Unknown')

    # Create per-email folder
    folder_name = re.sub(r'[^\w\s_]', '', subject).strip()[:50]
    email_path = os.path.join(base_dir, f"{msg_id}_{folder_name}")
    os.makedirs(email_path, exist_ok=True)

    images_dir = os.path.join(email_path, 'images')
    os.makedirs(images_dir, exist_ok=True)

    extracted_data = {
        'id':           msg_id,
        'subject':      subject,
        'sender':       sender,
        'date':         date,
        'text_plain':   '',
        'text_html':    '',
        'text_cleaned': '',
        'preheader':    '',
        'alt_texts':    [],
        'images':       [],
        'folder_path':  email_path
    }

    image_index = 0

    # -----------------------------------------------------------------------
    # RECURSIVE MIME PART PROCESSOR
    # -----------------------------------------------------------------------

    def process_parts(parts):
        nonlocal image_index
        plain_text = ""
        html_text  = ""

        for part in parts:
            filename      = part.get('filename', '')
            mime_type     = part.get('mimeType', '')
            body          = part.get('body', {})
            data          = body.get('data')
            attachment_id = body.get('attachmentId')

            # Plain text
            if mime_type == 'text/plain' and data:
                plain_text += base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')

            # HTML
            elif mime_type == 'text/html' and data:
                html_text += base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')

            # Inline or attachment images
            elif mime_type.startswith('image/'):
                image_index += 1
                ext = mime_type.split('/')[-1]
                if not filename:
                    filename = f"image_{image_index}.{ext}"

                image_path = os.path.join(images_dir, filename)

                try:
                    if attachment_id:
                        att       = service.users().messages().attachments().get(
                                        userId='me', messageId=msg_id, id=attachment_id
                                    ).execute()
                        file_data = base64.urlsafe_b64decode(att['data'].encode('UTF-8'))
                        source_type = 'attachment'
                    elif data:
                        file_data   = base64.urlsafe_b64decode(data)
                        source_type = 'inline'
                    else:
                        continue

                    with open(image_path, 'wb') as f:
                        f.write(file_data)

                    extracted_data['images'].append({
                        'image_index': image_index,
                        'filename':    filename,
                        'path':        image_path,
                        'source_url':  None,
                        'alt_text':    '',
                        'source_type': source_type,
                        'mime_type':   mime_type
                    })
                    print(f"  ✓ [{source_type}] Saved: {filename}")

                except Exception as e:
                    print(f"  ✗ Failed to save image {filename}: {e}")

            # Recurse for multipart
            elif 'parts' in part:
                sub_plain, sub_html = process_parts(part['parts'])
                plain_text += sub_plain
                html_text  += sub_html

        return plain_text, html_text

    # -----------------------------------------------------------------------
    # PROCESS EMAIL PARTS
    # -----------------------------------------------------------------------

    if 'parts' in payload:
        plain, html = process_parts(payload['parts'])
        extracted_data['text_plain'] = plain
        extracted_data['text_html']  = html
    else:
        data = payload.get('body', {}).get('data')
        if data:
            decoded   = base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
            mime_type = payload.get('mimeType', '')
            if mime_type == 'text/html':
                extracted_data['text_html']  = decoded
            else:
                extracted_data['text_plain'] = decoded

    # -----------------------------------------------------------------------
    # HTML POST-PROCESSING
    # -----------------------------------------------------------------------

    if extracted_data['text_html']:
        soup = BeautifulSoup(extracted_data['text_html'], 'html.parser')

        # --- Preheader: first hidden element = email summary (valuable!) ---
        preheader    = ''
        hidden_elems = soup.find_all(
            style=lambda v: v and ('display:none' in v.replace(' ', '') or
                                   'visibility:hidden' in v.replace(' ', ''))
        )
        if hidden_elems:
            preheader = hidden_elems[0].get_text(strip=True)
            extracted_data['preheader'] = preheader

        # Remove all hidden elements after capturing preheader
        for elem in hidden_elems:
            elem.decompose()

        # Remove noise tags
        for tag in soup(['script', 'style', 'meta', 'link']):
            tag.decompose()

        # --- Alt texts ---
        alt_texts = []
        for img in soup.find_all('img'):
            alt = img.get('alt', '').strip()
            if alt:
                alt_texts.append(alt)
        extracted_data['alt_texts'] = alt_texts

        # --- Download external images ---
        for idx, img in enumerate(soup.find_all('img'), start=100):
            src = img.get('src', '')
            alt = img.get('alt', '')

            # Skip non-http and tracking pixels
            if not src or not src.startswith('http'):
                continue
            if is_tracking_pixel(src, alt):
                print(f"  ⊘ Skipped tracking pixel: {src[:60]}...")
                continue

            image_index += 1
            try:
                resp = requests.get(src, timeout=10)
                if resp.status_code == 200:
                    parsed   = urlparse(src)
                    ext      = os.path.splitext(parsed.path)[1] or '.png'
                    filename = f"ext_image_{image_index}{ext}"
                    img_path = os.path.join(images_dir, filename)

                    with open(img_path, 'wb') as f:
                        f.write(resp.content)

                    extracted_data['images'].append({
                        'image_index': image_index,
                        'filename':    filename,
                        'path':        img_path,
                        'source_url':  src,
                        'alt_text':    alt,
                        'source_type': 'external',
                        'mime_type':   f"image/{ext.lstrip('.') or 'png'}"
                    })
                    print(f"  ✓ [external] Downloaded: {filename} | alt: '{alt[:40]}'")

            except Exception as e:
                print(f"  ✗ Failed to download external image {idx}: {e}")

        # --- Visible text extraction ---
        visible_text = soup.get_text(separator=' ', strip=True)

        # Clean invisible unicode characters
        for char in ['\u200c', '\u200b', '\u200d', '\ufeff',
                     '\u00a0', '\u2007', '\u202f', '͏']:
            visible_text = visible_text.replace(char, '')
        visible_text = re.sub(r'\s+', ' ', visible_text).strip()

        # --- Assemble unified text corpus ---
        text_parts = [
            subject,
            preheader,
            ' '.join(alt_texts),
            visible_text,
        ]
        extracted_data['text_cleaned'] = re.sub(
            r'\s+', ' ',
            ' '.join(t for t in text_parts if t)
        ).strip()

    else:
        extracted_data['text_cleaned'] = extracted_data['text_plain']

    # -----------------------------------------------------------------------
    # SAVE METADATA + CONTENT
    # -----------------------------------------------------------------------

    metadata_path = os.path.join(email_path, 'metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump({
            'id':         msg_id,
            'subject':    subject,
            'sender':     sender,
            'date':       date,
            'preheader':  extracted_data['preheader'],
            'alt_texts':  extracted_data['alt_texts'],
            'images':     extracted_data['images']
        }, f, indent=2)

    content_path = os.path.join(email_path, 'content.txt')
    with open(content_path, 'w', encoding='utf-8') as f:
        f.write(f"Subject:   {subject}\n")
        f.write(f"From:      {sender}\n")
        f.write(f"Date:      {date}\n")
        f.write(f"Preheader: {extracted_data['preheader']}\n")
        f.write(f"Alt Texts: {extracted_data['alt_texts']}\n")
        f.write(f"\n{'='*60}\n\n")
        f.write(extracted_data['text_cleaned'])

    print(f"\n  📁 Saved to: {email_path}")
    print(f"  🖼  Images ready for CLIP: {len(extracted_data['images'])}")

    return extracted_data


# ---------------------------------------------------------------------------
# FETCH MULTIPLE PROMOTIONAL EMAILS
# ---------------------------------------------------------------------------

def fetch_promotional_emails(service, max_results: int = 5) -> list:
    """
    Fetch and extract N promotional emails.
    Returns list of extracted_data dicts.
    """
    print(f"\nFetching {max_results} promotional emails...")

    results = service.users().messages().list(
        userId='me',
        q='category:promotions',
        maxResults=max_results
    ).execute()

    messages = results.get('messages', [])
    if not messages:
        print("No promotional emails found.")
        return []

    all_emails = []
    for i, msg in enumerate(messages, 1):
        print(f"\n[{i}/{len(messages)}] Processing email ID: {msg['id']}")
        email_data = extract_email_with_images(service, msg['id'])
        all_emails.append(email_data)

    return all_emails


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("="*60)
    print("GMAIL EXTRACTOR")
    print("="*60)

    service    = get_gmail_service()
    all_emails = fetch_promotional_emails(service, max_results=5)

    print("\n" + "="*60)
    print("EXTRACTION SUMMARY")
    print("="*60)
    for email in all_emails:
        print(f"\n  Subject: {email['subject'][:60]}")
        print(f"  Images:  {len(email['images'])}")
        print(f"  Text:    {len(email['text_cleaned'])} chars")
        print(f"  Folder:  {email['folder_path']}")

    print(f"\n✓ Total emails extracted: {len(all_emails)}")
    print(f"✓ Images ready for Dimension 1 CLIP indexing")