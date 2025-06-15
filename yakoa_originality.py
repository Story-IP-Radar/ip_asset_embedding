import json
import os
import time
import requests
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# === CONFIG === #
API_KEY = os.getenv("YAKOA_API_KEY")
NETWORK = os.getenv("YAKOA_NETWORK")
BASE_URL = f"https://{NETWORK}.ip-api-sandbox.yakoa.io/{NETWORK}"

INPUT_FILE = "captioned_assets.ndjson"
OUTPUT_FILE = "captioned_assets_with_yakoa.ndjson"

HEADERS = {
    "accept": "application/json",
    "content-type": "application/json",
    "X-API-KEY": API_KEY,
}

# === HELPERS === #

def get_token_id(record):
    chain = record["nftMetadata"]["chainId"]
    contract = record["nftMetadata"]["tokenContract"]
    token_id = record["nftMetadata"]["tokenId"]
    return f"{contract}:{token_id}", chain

def get_existing_ids():
    if not os.path.exists(OUTPUT_FILE):
        return set()
    with open(OUTPUT_FILE, "r") as f:
        return {json.loads(line)["id"] for line in f}

def register_token(token_id, chain, record):
    url = f"{BASE_URL}/{chain}/token"
    body = {
        "id": token_id,
        "registration_tx": {
            "block_number": int(record["blockNumber"]),
            "transaction_hash": record["transactionHash"],
            "timestamp": int(record["blockTimestamp"])
        },
        "creator_id": record["ipId"],
        "metadata": record["nftMetadata"],
        "media": [
            {
                "media_id": "default",
                "url": record["nftMetadata"]["imageUrl"]
            }
        ]
    }
    try:
        r = requests.post(url, headers=HEADERS, json=body, timeout=10)
        return r.status_code in (200, 201)
    except Exception as e:
        print(f"Register failed for {token_id}: {e}")
        return False

def get_token_status(token_id, chain):
    url = f"{BASE_URL}/{chain}/token/{token_id}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        print(f"GET token failed for {token_id}: {e}")
    return None

# === MAIN === #

existing_ids = get_existing_ids()

with open(INPUT_FILE, "r") as f_in, open(OUTPUT_FILE, "a") as f_out:
    for line in tqdm(f_in, desc="Checking assets"):
        record = json.loads(line)
        asset_id = record["id"]
        if asset_id in existing_ids:
            continue

        token_id, chain = get_token_id(record)
        token_status = get_token_status(token_id, chain)

        if not token_status:
            success = register_token(token_id, chain, record)
            if success:
                time.sleep(2)  # allow time for processing
                token_status = get_token_status(token_id, chain)

        if token_status:
            record["yakoa"] = {
                "infringements": token_status.get("infringements"),
                "originality_score": token_status.get("originality_score"),
                "hash_mismatch": any(
                    m.get("fetch_status") == "hash_mismatch"
                    for m in token_status.get("media", [])
                ),
            }

        f_out.write(json.dumps(record) + "\n")
        f_out.flush()
