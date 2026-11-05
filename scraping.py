import os
import re
import json
import time
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

CROP_INDEX_URLS = {
    "Tomato": "https://agritech.tnau.ac.in/crop_protection/crop_prot_crop%20diseases_veg_tomato.html",
    "Potato": "https://agritech.tnau.ac.in/crop_protection/crop_prot_crop%20diseases_veg_potato.html",
    "Rice": "https://agritech.tnau.ac.in/crop_protection/crop_prot_crop%20diseases_cereals_rice_main.html",
    "Maize": "https://agritech.tnau.ac.in/crop_protection/crop_prot_crop%20diseases_cereals_maize.html",
    "Banana": "https://agritech.tnau.ac.in/crop_protection/crop_prot_crop%20diseases_fruits_banana.html",
    "Grapes": "https://agritech.tnau.ac.in/crop_protection/crop_prot_crop%20diseases_fruits_grapes.html",
    "Chilli": "https://agritech.tnau.ac.in/crop_protection/crop_prot_crop%20diseases_veg_chilli.html",
    "Cotton": "https://agritech.tnau.ac.in/crop_protection/crop_prot_crop%20diseases_fibres_cotton.html"
}

NAV_BLACKLIST = {
    "home", "about us", "success stories", "farmers association", 
    "farmers' innovation", "farmers innovation", "publications", 
    "contact us", "contact", "disclaimer", "feedback", "back", "next", 
    "click here", "related links", "updated on"
}

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def scrape_disease_detail(crop: str, page_url: str, link_label: str) -> dict:
    """Extracts disease details and strips out standard site navigation text."""
    try:
        res = requests.get(page_url, headers=HEADERS, timeout=10)
        if res.status_code != 200:
            return None

        soup = BeautifulSoup(res.content, "lxml")

        for tag in soup.find_all(["nav", "footer", "header", "script", "style"]):
            tag.decompose()


        headers = soup.find_all(["h2", "h3", "h4", "strong", "b"])
        title = link_label
        for h in headers:
            text = clean_text(h.get_text())
            if text and len(text) > 3 and not any(nav in text.lower() for nav in NAV_BLACKLIST):
                title = text
                break

        full_text = soup.get_text(separator="\n")
        lines = [clean_text(l) for l in full_text.splitlines() if clean_text(l)]
        
        filtered_lines = [
            line for line in lines 
            if not any(nav == line.lower() or f"{nav} |" in line.lower() for nav in NAV_BLACKLIST)
        ]
        content = " ".join(filtered_lines)

        symptoms = ""
        symp_match = re.search(
            r'(?:Symptoms?|Field Diagnostic Symptoms?)(.*?)(?:Pathogen|Causal organism|Favourable condition|Survival|Mode of spread|Management|Integrated Disease Management|$)', 
            content, 
            re.IGNORECASE
        )
        if symp_match:
            symptoms = clean_text(symp_match.group(1))

        management = ""
        mgmt_match = re.search(
            r'(?:Management|Integrated Disease Management|Control measures)(.*?)(?:Source of Information|Acknowledgement|Content validator|Home|$)', 
            content, 
            re.IGNORECASE
        )
        if mgmt_match:
            management = clean_text(mgmt_match.group(1))
        else:

            management = content[:800]

        if len(management) < 25 and len(symptoms) < 25:
            return None

        return {
            "crop": crop,
            "disease": title,
            "url": page_url,
            "symptoms": symptoms if symptoms else "Refer to management instructions.",
            "management": management,
            "source": "TNAU Agritech Portal"
        }

    except Exception as e:
        print(f"    [-] Error on {page_url}: {e}")
        return None

def run_scraper(output_file="tnau_knowledge.json"):
    all_diseases = []
    print("[*] Starting TNAU Disease Knowledge Extraction...\n")

    for crop, index_url in CROP_INDEX_URLS.items():
        print(f"[+] Processing {crop}...")
        try:
            res = requests.get(index_url, headers=HEADERS, timeout=12)
            if res.status_code != 200:
                print(f"    [-] Failed to reach {crop} directory ({res.status_code})")
                continue

            soup = BeautifulSoup(res.content, "lxml")
            links = soup.find_all("a", href=True)
            
            valid_disease_links = []
            seen_urls = set()

            for a in links:
                href = a["href"].strip()
                label = clean_text(a.get_text())

                if not label or len(label) < 3:
                    continue
                if any(nav in label.lower() for nav in NAV_BLACKLIST):
                    continue
                if href.startswith("http") and "agritech.tnau.ac.in" not in href:
                    continue
                if any(x in href.lower() for x in ["contact", "about", "success", "publication", "index", "home"]):
                    continue

                full_url = urljoin(index_url, href)
                if full_url != index_url and full_url not in seen_urls:
                    seen_urls.add(full_url)
                    valid_disease_links.append((full_url, label))

            print(f"    Found {len(valid_disease_links)} disease links.")

            crop_count = 0
            for url, label in valid_disease_links:
                data = scrape_disease_detail(crop, url, label)
                if data:
                    all_diseases.append(data)
                    crop_count += 1
                    print(f"      [✓] {data['disease'][:40]}")
                time.sleep(0.3)

            print(f"    Saved {crop_count} records for {crop}.\n")

        except Exception as e:
            print(f"    [-] Failed for {crop}: {e}\n")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_diseases, f, indent=2, ensure_ascii=False)

    print(f"[✓] Extraction Complete! Total verified diseases indexed: {len(all_diseases)}")
    print(f"[✓] Output written to {output_file}")

if __name__ == "__main__":
    run_scraper()
