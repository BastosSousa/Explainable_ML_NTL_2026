# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 12:34:04 2025

@author: Natalia
"""
import os
from googlesearch import search
import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin
import pandas as pd

from urllib.parse import urljoin, urlparse
#%%
def find_dso_url(dso_name, num_results=5):
    query = dso_name
    print(f"Searching for '{dso_name}'…")
    for url in search(query, lang="de", num_results=num_results):
        print(f"  Checking: {url}")
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                title = BeautifulSoup(resp.text, "html.parser").title
                if title and dso_name.lower().split()[0] in title.text.lower():
                    print(f" → Found: {url}")
                    return url
        except Exception as e:
            print(f"   Error loading {url}: {e}")
    print("No good match found.")
    return None

#%%

input_file = r'C:\Pos\germancase\germanDSOs.csv'

df = pd.read_csv(input_file)
#%%

for i in range(len(df)):
    print(df['Name des Marktakteurs'][i])
    dso_value = df['Name des Marktakteurs'][i]
    dso = dso_value
    url = find_dso_url(dso)
    df['URL'][i] =  url
    print("Result:", url)


df.to_csv('germanDSOs_out.csv', index=False)  

#%%


def crawl_for_netzverluste(base_url, max_depth=3, download_dir="downloads"):
    visited = set()
    found_files = []

    os.makedirs(download_dir, exist_ok=True)

    def crawl(url, depth):
        if depth > max_depth or url in visited:
            return
        visited.add(url)

        try:
            resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=120)
        except Exception as e:
            print(f"❌ Failed to load {url}: {e}")
            return

        if resp.status_code != 200 or "text/html" not in resp.headers.get("content-type", ""):
            return

        soup = BeautifulSoup(resp.text, "html.parser")
        domain = urlparse(base_url).netloc

        for link in soup.find_all("a", href=True):
            href = link["href"]
            link_text = link.get_text(strip=True).lower()
            full_url = urljoin(url, href)

            # Only crawl internal links
            if urlparse(full_url).netloc == domain:
                # If "Netzverluste" in text or file link → download
                if "netzverluste" in link_text or "netzverluste" in href.lower():
                    if full_url.endswith(".pdf"):
                        filename = os.path.basename(full_url.split("?")[0])
                        save_path = os.path.join(download_dir, filename)
                        try:
                            print(f"⬇️ Downloading {full_url}")
                            file_resp = requests.get(full_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=120)
                            if file_resp.status_code == 200:
                                with open(save_path, "wb") as f:
                                    f.write(file_resp.content)
                                print(f"✅ Saved: {save_path}")
                                found_files.append(save_path)
                        except Exception as e:
                            print(f"❌ Error downloading {full_url}: {e}")
                else:
                    # Continue crawling deeper
                    crawl(full_url, depth + 1)

    crawl(base_url, 0)
    return found_files

#%%

if __name__ == "__main__":
    
    
    for i in range(len(df)):
        url = df['URL'][i]
        if url != None:
            docs = crawl_for_netzverluste(url, max_depth=2)
            if docs:
                df['Tätigkeitsende'][i] = "Yes"
                print("\n📂 Collected Netzverluste documents:")
            else:
                df['Tätigkeitsende'][i] = "No"
                print("\n⚠️ No 'Netzverluste' docs found.")
                
    df.to_csv('germanDSOs_out2.csv', index=False)  
