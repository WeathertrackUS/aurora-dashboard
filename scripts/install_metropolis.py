"""
Small helper to download Metropolis font files into static/fonts/metropolis/

Usage:
    python scripts/install_metropolis.py --url <font_zip_or_folder_url>

If no URL is supplied the script will attempt a few known public forks. If
it cannot find fonts, it will instruct you where to place font files.

You must confirm licensing before using the font in production.
"""
import os
import sys
import argparse
import shutil
import requests
from zipfile import ZipFile
from io import BytesIO

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'static', 'fonts', 'metropolis')
OUT_DIR = os.path.normpath(OUT_DIR)

CANDIDATE_ZIPS = [
    # Public forks that historically hosted Metropolis - may change or be removed
    'https://github.com/madebybowtie/Metropolis/archive/refs/heads/master.zip',
    'https://github.com/matthieua/Metropolis/archive/refs/heads/master.zip'
]


def ensure_out_dir():
    os.makedirs(OUT_DIR, exist_ok=True)


def extract_fonts_from_zip(content):
    z = ZipFile(BytesIO(content))
    extracted = 0
    for name in z.namelist():
        lower = name.lower()
        if lower.endswith('.ttf') or lower.endswith('.otf') or lower.endswith('.woff') or lower.endswith('.woff2'):
            dest_name = os.path.basename(name)
            dest_path = os.path.join(OUT_DIR, dest_name)
            print('Extracting', name, '->', dest_path)
            with z.open(name) as src, open(dest_path, 'wb') as dst:
                shutil.copyfileobj(src, dst)
            extracted += 1
    return extracted


def try_urls(urls):
    for url in urls:
        print('Trying', url)
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            print('Downloaded', len(r.content), 'bytes, extracting...')
            n = extract_fonts_from_zip(r.content)
            if n:
                print('Extracted', n, 'font files to', OUT_DIR)
                return True
        except Exception as e:
            print('Failed:', e)
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', help='Zip URL or direct font file URL to download')
    args = parser.parse_args()

    ensure_out_dir()

    if args.url:
        url = args.url
        print('Attempting user-supplied URL:', url)
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            ct = r.headers.get('content-type','')
            if 'zip' in ct or url.lower().endswith('.zip'):
                n = extract_fonts_from_zip(r.content)
                if n:
                    print('Extracted', n, 'fonts into', OUT_DIR)
                    return
            else:
                # Save single file
                fname = os.path.basename(url.split('?')[0])
                dest = os.path.join(OUT_DIR, fname)
                with open(dest, 'wb') as f:
                    f.write(r.content)
                print('Saved single font to', dest)
                return
        except Exception as e:
            print('Failed to download from URL:', e)

    print('No user URL or failed download, trying known locations...')
    ok = try_urls(CANDIDATE_ZIPS)
    if ok:
        print('Fonts installed. Restart the server to pick them up.')
    else:
        print('\nCould not automatically obtain Metropolis fonts.')
        print('Please download Metropolis font files (WOFF2/WOFF/TTF) yourself and place them in:')
        print('  ', OUT_DIR)
        print('Then restart the server. Be sure you have the right to use the font.')

if __name__ == '__main__':
    main()
