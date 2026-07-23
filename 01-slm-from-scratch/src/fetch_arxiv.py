import re 
import xml.etree.ElementTree as ET
import urllib.request
import json, time, os

NS = {"atom": "http://www.w3.org/2005/Atom"} # arXiv's feed uses the Atom namespace

def clean(text:str) -> str:
    """
    Clean strings before storage into JSONL
    
    input: 
        text: str
            Text to be processed
            
    ouput:
        text: str
            Processed text (cleaned)
    """
    # collapse whitespace runs into one space, trim the ends
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text
    
def parse_entries(xml:str) -> list[tuple[str, str, str]]:
    """
    Parse the arXiv API's Atom XML feed into cleaned (id, title, abstract) tuples
    
    Extracts <id>, <title>, and <summary> from each <entry>; all other metadata 
    (authors, dates, categories, links) is dropped. Title and abstract are 
    whitespace-normalized via clean().
    
    input:
        xml: str
            The raw Atom XML response returned by the arXiv API
            
    output:
        out: list[tuple[str, str, str]]
            One (arxiv_id, title, abstract) tuple per paper.
    """
    
    root = ET.fromstring(xml)
    out = []
    for entry in root.findall("atom:entry", NS):
        arxiv_id = entry.find("atom:id", NS).text.rsplit("/",1)[-1]
        title = clean(entry.find("atom:title", NS).text)
        abstract = clean(entry.find("atom:summary", NS).text)
        out.append((arxiv_id, title, abstract))
    return out 

def fetch_batch(category:str, start:int, n:int) -> str:
    """
    Build the URL and send the request to the arXiv API.
    
    input:
        category:
            arXiv subject: "cs.LG" -> Machine Learning
            
        start:
            Start index in the result list
            
        n:
            Number of results to return
            
    out:
        Returns decoded utf-8 str from arXiv API
    """
    
    url = f"http://export.arxiv.org/api/query?search_query=cat:{category}&start={start}&max_results={n}"
    xml = urllib.request.urlopen(url, timeout=30).read().decode("utf-8")
    return xml

def build_dataset(category: str, target: int, batch_size: int = 100, out_dir: str = "data") -> int:
    """
    Download ´target´ (id, title, abstract) pairs for ´category´ into JSONL + a manifest.
    
    input:
        category:
            arXiv subject: "cs.LG" -> Machine Learning
        target:
            target
        batch_size:
            batch size, default = 100
        out_dir:
            output directory, default = "data"
            
    output:
        int
    """
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"arxiv_{category.replace('.', '_')}.jsonl")
    ids_path  = os.path.join(out_dir, "paper_ids.txt")
    
    # Check existing papers
    paper_ids_path = os.path.join(out_dir, "paper_ids.txt")
    if os.path.exists(paper_ids_path):
        with open (paper_ids_path, "r", encoding="utf-8") as f:
            seen = set(line.strip() for line in f)
            count = len(seen)
    else:
        seen = set()
        count = 0
        
    start = 0
    with open(json_path, "a", encoding="utf-8") as jf, open(ids_path, "a", encoding="utf-8") as idf:
        while count < target:
            entries = parse_entries(fetch_batch(category, start, batch_size))
            if not entries:
                break
            for arxiv_id, title, abstract in entries:
                if arxiv_id in seen:
                    continue
                seen.add(arxiv_id)
                jf.write(json.dumps({"id": arxiv_id, "title": title, "abstract": abstract})+"\n")
                idf.write(arxiv_id + "\n")
                count += 1
                if count >= target:
                    break
            start += batch_size
            time.sleep(3)
    return count

def restore_from_manifest(manifest_path: str = "data/paper_ids.txt", out_dir: str = "data", out_name: str = "arxiv_cs_LG.jsonl", batch_size: int = 100) -> int:
    """
    Rebuilds the JSONL from the committed id manifest - reproducible in any machine
    """
    with open(manifest_path, encoding="utf-8") as f:
        ids = [line.strip() for line in f if line.strip()]
        
    json_path = os.path.join(out_dir, out_name)
    with open(json_path, "w", encoding="utf-8") as jf:
        for i in range(0, len(ids), batch_size):
            batch = ids[i:i + batch_size]
            for arxiv_id, title, abstract in parse_entries(fetch_by_ids(batch)):
                jf.write(json.dumps({"id": arxiv_id, "title": title, "abstract": abstract}) + "\n")
            time.sleep(3)
    return len(ids)

def fetch_by_ids(ids: list[str]) -> str:
    """Fetch specific papers from arXiv by their ids (via the id_list parameter)."""
    url = f"http://export.arxiv.org/api/query?id_list={','.join(ids)}&max_results={len(ids)}"
    xml = urllib.request.urlopen(url, timeout=30).read().decode("utf-8")
    return xml
