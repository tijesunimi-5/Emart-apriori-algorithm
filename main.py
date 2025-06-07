import os
import json
from pymongo import MongoClient
from apyori import apriori
from fastapi import FastAPI, HTTPException
import uvicorn
import asyncio
from pymongo.errors import PyMongoError
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI()

# CORS configuration for Next.js
origins = [
    "https://e-mart-rho.vercel.app",
    "http://localhost:3000",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration with environment variables
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb+srv://tijesunimiidowu16:M7UN0QTHvX6P5ktw@cluster0.x5257.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
# Use platform-specific default paths with explicit environment override
if os.name == "nt":  # Windows
    OUTPUT_DIR = os.getenv("OUTPUT_DIR", "C:/Users/Admin/OneDrive/Desktop/Model/tensorflow/ipynb/challenges/Real_World_projects/emart_apriori_backend/rules")
else:  # Linux (Render)
    OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/emart_apriori_backend/rules")  # Match disk mount
RULES_FILE_PATH = os.path.join(OUTPUT_DIR, "apriori_rules.json")
print(f"DEBUG: Running on platform: {os.name}", flush=True)
print(f"DEBUG: Configured OUTPUT_DIR: {OUTPUT_DIR}", flush=True)
print(f"DEBUG: Expected RULES_FILE_PATH: {RULES_FILE_PATH}", flush=True)

# Ensure output directory exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"DEBUG: Created missing OUTPUT_DIR: {OUTPUT_DIR}", flush=True)
else:
    print(f"DEBUG: OUTPUT_DIR already exists: {OUTPUT_DIR}", flush=True)

# MongoDB connection
client = MongoClient(MONGODB_URI)
db = client["ecommerce"]
transactions_collection = db["transactions"]
print(f"DEBUG: Using MONGODB_URI: {MONGODB_URI}", flush=True)

# Fetch transactions
def fetch_transactions():
    try:
        transactions_data = transactions_collection.find({}, {"items": 1, "_id": 0})
        transactions = [[item["productId"] for item in tx["items"]] for tx in transactions_data if tx.get("items")]
        print(f"Fetched transactions: {len(transactions)} transactions for Apriori", flush=True)
        return transactions
    except PyMongoError as e:
        print(f"ERROR: Failed to fetch transactions: {e}", flush=True)
        return []

# Update Apriori rules
def update_apriori_rules():
    transactions = fetch_transactions()
    if not transactions:
        print("WARNING: No transactions fetched", flush=True)
        return

    rules = apriori(
        transactions,
        min_support=0.001,
        min_confidence=0.2,
        min_lift=1.0,
        min_length=2
    )

    rules_list = []
    for rule in list(rules):
        for ordered_stat in rule.ordered_statistics:
            rules_list.append({
                "antecedents": list(ordered_stat.items_base),
                "consequents": list(ordered_stat.items_add),
                "support": rule.support,
                "confidence": ordered_stat.confidence,
                "lift": ordered_stat.lift
            })

    with open(RULES_FILE_PATH, "w") as file:
        json.dump(rules_list, file, indent=2)
    print(f"Updated {RULES_FILE_PATH} with {len(rules_list)} rules", flush=True)

# Watch transactions
async def watch_transactions():
    print("Starting MongoDB Change Stream to watch for new transactions...", flush=True)
    rules_generated = False
    previous_user_id = None
    try:
        with transactions_collection.watch() as stream:
            async for change in stream:
                print("Change detected:", change, flush=True)
                if change["operationType"] in ["insert", "update"]:
                    print("New transaction detected...", flush=True)
                    transaction = transactions_collection.find_one({"_id": change["documentKey"]["_id"]})
                    if transaction and "userId" in transaction:
                        current_user_id = transaction["userId"]
                        if not rules_generated:
                            print("Generating initial Apriori rules...", flush=True)
                            update_apriori_rules()
                            rules_generated = True
                        elif current_user_id != previous_user_id:
                            print("New cart session detected, regenerating Apriori rules...", flush=True)
                            update_apriori_rules()
                        previous_user_id = current_user_id
    except PyMongoError as e:
        print(f"ERROR in Change Stream: {e}", flush=True)
    except Exception as e:
        print(f"Unexpected error in Change Stream: {e}", flush=True)
    finally:
        client.close()

# API Endpoints
@app.get("/api/rules")
async def get_rules():
    print(f"GET /api/rules endpoint hit. Checking {RULES_FILE_PATH}", flush=True)
    if not os.path.exists(RULES_FILE_PATH):
        print(f"DEBUG: RULES_FILE_PATH does NOT exist: {RULES_FILE_PATH}", flush=True)
        raise HTTPException(status_code=404, detail="Rules not yet generated")
    file_size = os.path.getsize(RULES_FILE_PATH)
    print(f"DEBUG: RULES_FILE_PATH exists with size: {file_size} bytes", flush=True)
    try:
        with open(RULES_FILE_PATH, "r") as file:
            rules_content = file.read()
            if not rules_content.strip():
                print(f"DEBUG: RULES_FILE_PATH is empty or only whitespace.", flush=True)
                raise HTTPException(status_code=404, detail="Rules file is empty")
            rules = json.loads(rules_content)
            print(f"DEBUG: Successfully parsed {len(rules)} rules.", flush=True)
            return rules
    except json.JSONDecodeError as e:
        print(f"ERROR: JSON Decode Error reading {RULES_FILE_PATH}: {e}", flush=True)
        raise HTTPException(status_code=500, detail=f"Failed to parse rules file: Invalid JSON: {e}")
    except Exception as e:
        print(f"ERROR: Generic error in get_rules: {e}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/update-rules")
async def update_rules():
    print("POST /api/update-rules endpoint hit. Triggering rule update...", flush=True)
    update_apriori_rules()
    return {"message": "Rules updated successfully"}

# Main function with initial rule generation
async def main():
    print("Forcing initial rule generation on startup...", flush=True)
    update_apriori_rules()  # Force initial rule generation
    asyncio.create_task(watch_transactions())
    config = uvicorn.Config(app, host="0.0.0.0", port=8000)
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main())