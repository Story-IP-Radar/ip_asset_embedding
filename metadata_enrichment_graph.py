import os
import json
import time
import re
import requests
from typing import Optional, Dict, List, TypedDict
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, END

OLLAMA_MODEL = "llama3"
MAX_RETRIES = 3
IPFS_GATEWAY = "https://gateway.pinata.cloud/ipfs/"
MODEL = SentenceTransformer("all-MiniLM-L6-v2")
CHECKPOINT_EVERY = 100
CHECKPOINT_FILE = "checkpoint.txt"
VECTOR_OUTPUT_FILE = "enriched_vectors.jsonl"
FAILURE_LOG = "failures.log"

class AssetState(TypedDict, total=False):
    id: str
    name: Optional[str]
    imageCaption: Optional[str]
    tokenUri: Optional[str]
    normalizedUri: Optional[str]
    fetchedMetadata: Optional[dict]
    fieldsToUse: Optional[List[str]]
    descriptionText: Optional[str]
    complete: bool

def normalize_token_uri(uri: Optional[str]) -> Optional[str]:
    if not uri:
        return None
    if uri.startswith("ipfs://"):
        return uri.replace("ipfs://", IPFS_GATEWAY)
    m = re.search(r"ipfs/([\w\d]+.*)", uri)
    if m:
        return IPFS_GATEWAY + m.group(1)
    if uri.startswith("http"):
        return uri
    return None

def fetch_json(uri: str, retries: int = MAX_RETRIES) -> Optional[Dict]:
    for _ in range(retries):
        try:
            r = requests.get(uri, timeout=10)
            if r.status_code == 200 and r.headers.get("Content-Type","").startswith("application/json"):
                return r.json()
        except:
            time.sleep(1)
    return None

def extract_keys_via_llm(json_data: Dict) -> List[str]:
    prompt = "You are a data extraction assistant. Given a JSON NFT metadata, return only field names useful for description."
    try:
        res = requests.post("http://localhost:11434/api/generate", json={
            "model": OLLAMA_MODEL,
            "prompt": prompt + "\n" + json.dumps(json_data, indent=2),
            "system": "",
            "stream": False
        }, timeout=30)
        res.raise_for_status()
        lines = res.json().get("response","").splitlines()
        return [ln.strip("- ").strip() for ln in lines if ln.strip()]
    except:
        return []

def flatten_and_extract(metadata: Dict, fields: List[str]) -> str:
    vals = []
    for f in fields:
        v = metadata.get(f)
        if isinstance(v, str):
            vals.append(v)
        elif isinstance(v, list):
            for x in v:
                if isinstance(x, dict):
                    vals.extend(map(str, x.values()))
                else:
                    vals.append(str(x))
        elif isinstance(v, dict):
            vals.extend(map(str, v.values()))
    return ". ".join(vals)

def inspect_token_uri(state: AssetState) -> AssetState:
    uri = state.get("tokenUri")
    state["normalizedUri"] = normalize_token_uri(uri)
    if uri and not state["normalizedUri"]:
        state["descriptionText"] = uri
        state["complete"] = True
    return state

def fetch_metadata(state: AssetState) -> AssetState:
    if state.get("complete"):
        return state
    uri = state.get("normalizedUri")
    metadata = fetch_json(uri) if uri else None
    if metadata:
        state["fetchedMetadata"] = metadata
    else:
        state["descriptionText"] = "Unfetchable metadata"
        state["complete"] = True
    return state

def run_llm_extraction(state: AssetState) -> AssetState:
    if "fetchedMetadata" not in state:
        state["descriptionText"] = "No metadata available"
        state["complete"] = True
        return state
    state["fieldsToUse"] = extract_keys_via_llm(state["fetchedMetadata"])
    return state

def extract_description(state: AssetState) -> AssetState:
    name = state.get("name","") or ""
    caption = state.get("imageCaption","") or ""
    meta = state.get("fetchedMetadata", {})
    fields = state.get("fieldsToUse", [])
    txt = flatten_and_extract(meta, fields)
    combined = ". ".join([name, caption, txt])[:2000]
    state["descriptionText"] = combined
    return state

def vectorize_description(state: AssetState) -> AssetState:
    embed = MODEL.encode(state.get("descriptionText",""))
    out = {"id": state["id"], "descriptionText": state["descriptionText"], "embedding": embed.tolist()}
    with open(VECTOR_OUTPUT_FILE,"a") as o:
        o.write(json.dumps(out)+"\n")
    state["complete"] = True
    return state

def failure_logger(state: AssetState) -> AssetState:
    with open(FAILURE_LOG,"a") as f:
        f.write(json.dumps({"id": state["id"], "reason": state.get("descriptionText")}) + "\n")
    return state

# === BUILD GRAPH === #
graph = StateGraph(AssetState)
graph.add_node("inspect_token_uri", inspect_token_uri)
graph.add_node("fetch_metadata", fetch_metadata)
graph.add_node("run_llm_extraction", run_llm_extraction)
graph.add_node("extract_description", extract_description)
graph.add_node("vectorize_description", vectorize_description)
graph.add_node("failure_logger", failure_logger)

graph.set_entry_point("inspect_token_uri")
graph.add_edge("inspect_token_uri", "fetch_metadata")
graph.add_edge("fetch_metadata", "run_llm_extraction")
graph.add_edge("run_llm_extraction", "extract_description")
graph.add_edge("extract_description", "vectorize_description")
graph.add_conditional_edges("vectorize_description",
    lambda st: "failure_logger" if not st.get("complete") else END
)
graph.add_edge("failure_logger", END)

compiled = graph.compile()

# === RUN === #
def load_existing_ids() -> set:
    if not os.path.exists(VECTOR_OUTPUT_FILE):
        return set()
    with open(VECTOR_OUTPUT_FILE) as f:
        return {json.loads(line)["id"] for line in f}

if __name__ == "__main__":
    processed_ids = load_existing_ids()
    count = 0

    with open("captioned_assets.ndjson") as f:
        for line in f:
            rec = json.loads(line)
            if rec["id"] in processed_ids:
                continue
            state: AssetState = {
                "id": rec["id"],
                "tokenUri": rec.get("nftMetadata",{}).get("tokenUri"),
                "name": rec.get("nftMetadata",{}).get("name",""),
                "imageCaption": rec.get("imageCaption",""),
            }
            compiled.invoke(state)
            count += 1
            if count % CHECKPOINT_EVERY == 0:
                with open(CHECKPOINT_FILE, "w") as ck:
                    ck.write(rec["id"] + "\n")
