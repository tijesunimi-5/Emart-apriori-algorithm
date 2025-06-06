import pandas as pd
from pymongo import MongoClient
from apyori import apriori
import json
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import threading
import uvicorn
from dotenv import load_dotenv
import time

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for Vercel frontend and local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",         # For your local development frontend
        "https://e-mart-rho.vercel.app"  # For your deployed Vercel frontend
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB Connection
MONGODB_URI = os.getenv("MONGODB_URI")
client = MongoClient(MONGODB_URI)
db = client["ecommerce"]
transactions_collection = db["transactions"]

# Output directory
# This defaults to /app/rules if OUTPUT_DIR env var is not set
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/app/rules")
RULES_FILE_PATH = os.path.join(OUTPUT_DIR, "apriori_rules.json")

# --- DEBUGGING LOGS FOR FILE SYSTEM ---
# These will run when the app starts up
print(f"DEBUG: Application root path: {os.getcwd()}", flush=True) # Current working directory
print(f"DEBUG: Configured OUTPUT_DIR: {OUTPUT_DIR}", flush=True)
print(f"DEBUG: Expected RULES_FILE_PATH: {RULES_FILE_PATH}", flush=True)

# Ensure the output directory exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"DEBUG: Created missing OUTPUT_DIR: {OUTPUT_DIR}", flush=True)
else:
    print(f"DEBUG: OUTPUT_DIR already exists: {OUTPUT_DIR}", flush=True)
# --- END DEBUGGING LOGS ---

def fetch_transactions():
    transactions_data = transactions_collection.find({}, {"items": 1, "_id": 0})
    transactions = [[item['productId'] for item in tx['items']] for tx in transactions_data]
    print(f"Fetched transactions: {len(transactions)} transactions for Apriori", flush=True)
    return transactions

def update_apriori_rules():
    transactions = fetch_transactions()
    if not transactions:
        print("WARNING: No transactions fetched. Apriori will generate no rules.", flush=True)
        rules_list = [] # Ensure an empty list is written if no transactions
    else:
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
                    'antecedents': list(ordered_stat.items_base),
                    'consequents': list(ordered_stat.items_add),
                    'support': rule.support,
                    'confidence': ordered_stat.confidence,
                    'lift': ordered_stat.lift
                })
    
    with open(RULES_FILE_PATH, 'w') as file:
        json.dump(rules_list, file, indent=2)
    print(f"Updated {RULES_FILE_PATH} with {len(rules_list)} rules", flush=True)
    return rules_list

def watch_transactions():
    print("Starting MongoDB Change Stream to watch for new transactions...", flush=True)
    rules_generated = False
    previous_user_id = None
    while True:
        try:
            # Setting a cursor type might be helpful for initial full document retrieval if needed
            with transactions_collection.watch(full_document='updateLookup') as stream:
                for change in stream:
                    print(f"Change detected: Operation Type = {change.get('operationType')}", flush=True)
                    if change['operationType'] in ['insert', 'update']:
                        print("New transaction detected, checking for rule regeneration trigger...", flush=True)
                        if not rules_generated:
                            print("FIRST transaction detected, generating initial Apriori rules...", flush=True)
                            update_apriori_rules()
                            rules_generated = True
                        else:
                            transaction = transactions_collection.find_one({"_id": change["documentKey"]["_id"]})
                            current_user_id = transaction.get("userId") if transaction else None
                            if current_user_id and current_user_id != previous_user_id:
                                print(f"New cart session ({current_user_id}) detected, regenerating Apriori rules...", flush=True)
                                update_apriori_rules()
                            previous_user_id = current_user_id
        except Exception as e:
            print(f"ERROR: Change Stream error: {e}. Retrying in 5 seconds...", flush=True)
            time.sleep(5)

# Add root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the E-Mart Apriori API. Use /api/rules to get rules or /api/update-rules to update them."}

@app.get("/api/rules")
async def get_rules():
    try:
        # --- DEBUGGING LOGS FOR GET_RULES ---
        print(f"GET /api/rules endpoint hit. Checking {RULES_FILE_PATH}", flush=True)
        if not os.path.exists(RULES_FILE_PATH):
            print(f"DEBUG: RULES_FILE_PATH does NOT exist: {RULES_FILE_PATH}", flush=True)
            return {"error": "Rules not yet generated"}, 404
        
        file_size = os.path.getsize(RULES_FILE_PATH)
        print(f"DEBUG: RULES_FILE_PATH exists with size: {file_size} bytes", flush=True)

        with open(RULES_FILE_PATH, 'r') as file:
            rules_content = file.read()
            if not rules_content.strip():
                print(f"DEBUG: RULES_FILE_PATH is empty or only whitespace.", flush=True)
                # You might choose to return 404 or an empty array [] here
                return {"error": "Rules file is empty"}, 404 # Frontend expects an array, so this might still break it
            
            # Print a snippet of content if it's large, otherwise full content
            content_snippet = rules_content[:200] + "..." if len(rules_content) > 200 else rules_content
            print(f"DEBUG: Attempting to parse JSON from content: {content_snippet}", flush=True)
            
            rules = json.loads(rules_content)
            print(f"DEBUG: Successfully parsed {len(rules)} rules.", flush=True)
        # --- END DEBUGGING LOGS ---
        
        return rules
    except json.JSONDecodeError as e:
        print(f"ERROR: JSON Decode Error reading {RULES_FILE_PATH}: {e}", flush=True)
        return {"error": f"Failed to parse rules file: Invalid JSON: {e}"}, 500
    except Exception as e:
        print(f"ERROR: Generic error in get_rules: {e}", flush=True)
        return {"error": str(e)}, 500

@app.post("/api/update-rules")
async def trigger_update():
    print("POST /api/update-rules endpoint hit. Triggering rule update...", flush=True)
    try:
        rules = update_apriori_rules()
        print("Rules updated successfully via /api/update-rules.", flush=True)
        return {"message": "Rules updated successfully", "rules": rules}, 200
    except Exception as e:
        print(f"ERROR: Failed to update rules via /api/update-rules: {e}", flush=True)
        return {"error": str(e)}, 500

def start_background_thread():
    thread = threading.Thread(target=watch_transactions)
    thread.daemon = True
    thread.start()

if __name__ == "__main__":
    print("Application starting...", flush=True)
    # --- IMPORTANT: Ensure initial rule generation happens ---
    try:
        # This part ensures rules are generated immediately if the file doesn't exist
        # or if it's empty, regardless of change stream activity.
        if not os.path.exists(RULES_FILE_PATH) or os.path.getsize(RULES_FILE_PATH) == 0:
            print(f"Initial check: Rules file '{RULES_FILE_PATH}' missing or empty. Attempting to generate now.", flush=True)
            update_apriori_rules()
        else:
            print(f"Initial check: Rules file '{RULES_FILE_PATH}' already exists and is populated. Skipping initial generation.", flush=True)
    except Exception as e:
        print(f"ERROR: Critical error during initial rule generation on startup: {e}", flush=True)

    start_background_thread()
    print("Background transaction watcher thread started.", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))