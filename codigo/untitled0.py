# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 16:07:06 2026

@author: Natalia
"""

import pandas as pd
import requests
import time

#%%
df = pd.DataFrame({
    "cidade": ["Berlin", "Munich", "Hamburg", "Frankfurt"]
})

#%%


def geocode_city(city, country="Germany"):
    url = "https://nominatim.openstreetmap.org/search"
    
    params = {
        "q": f"{city}, {country}",
        "format": "json",
        "limit": 1
    }
    
    headers = {
        "User-Agent": "seu_app_nome (seu_email@exemplo.com)"
    }
    
    response = requests.get(url, params=params, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        if len(data) > 0:
            return float(data[0]["lat"]), float(data[0]["lon"])
    
    return None, None


#%%

lats = []
lons = []

for city in df["cidade"]:
    lat, lon = geocode_city(city)
    lats.append(lat)
    lons.append(lon)
    time.sleep(1)  # respeita o limite do Nominatim

#%%
df["latitude"] = lats
df["longitude"] = lons

#%%

print(df)
