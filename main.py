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

# Enable CORS for Vercel frontend
app.add_middleware(
    CORSMiddleware,
    # Directly list all allowed origins here
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
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/app/rules")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
RULES_FILE_PATH = os.path.join(OUTPUT_DIR, "apriori_rules.json")

def fetch_transactions():
    transactions_data = transactions_collection.find({}, {"items": 1, "_id": 0})
    transactions = [[item['productId'] for item in tx['items']] for tx in transactions_data]
    print(f"Fetched transactions: {transactions}")
    return transactions

def update_apriori_rules():
    transactions = fetch_transactions()
    print(f"Fetched {len(transactions)} transactions for Apriori")
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
    print(f"Updated {RULES_FILE_PATH} with {len(rules_list)} rules")
    return rules_list

def watch_transactions():
    print("Starting MongoDB Change Stream to watch for new transactions...")
    rules_generated = False
    previous_user_id = None
    while True:
        try:
            with transactions_collection.watch() as stream:
                for change in stream:
                    print("Change detected:", change)
                    if change['operationType'] in ['insert', 'update']:
                        print("New transaction detected...")
                        if not rules_generated:
                            print("Generating initial Apriori rules...")
                            update_apriori_rules()
                            rules_generated = True
                        else:
                            transaction = transactions_collection.find_one({"_id": change["documentKey"]["_id"]})
                            if transaction and "userId" in transaction and transaction["userId"] != previous_user_id:
                                print("New cart session detected, regenerating Apriori rules...")
                                update_apriori_rules()
                            previous_user_id = transaction.get("userId") if transaction else None
        except Exception as e:
            print(f"Change Stream error: {e}. Retrying in 5 seconds...")
            time.sleep(5)

# Add root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the E-Mart Apriori API. Use /api/rules to get rules or /api/update-rules to update them."}

@app.get("/api/rules")
async def get_rules():
    try:
        with open(RULES_FILE_PATH, 'r') as file:
            rules = json.load(file)
        return rules
    except FileNotFoundError:
        return {"error": "Rules not yet generated"}, 404
    except Exception as e:
        return {"error": str(e)}, 500

@app.post("/api/update-rules")
async def trigger_update():
    try:
        rules = update_apriori_rules()
        return {"message": "Rules updated successfully", "rules": rules}, 200
    except Exception as e:
        return {"error": str(e)}, 500

def start_background_thread():
    thread = threading.Thread(target=watch_transactions)
    thread.daemon = True
    thread.start()

if __name__ == "__main__":
    # --- ADD THIS BLOCK ---
    print("Attempting to generate initial Apriori rules on startup...")
    try:
        # Check if rules file exists or is empty before trying to generate
        if not os.path.exists(RULES_FILE_PATH) or os.path.getsize(RULES_FILE_PATH) == 0:
            print(f"'{RULES_FILE_PATH}' not found or is empty. Generating rules now.")
            update_apriori_rules()
        else:
            # Optionally, you could still regenerate here if you want fresh rules on every deploy
            # update_apriori_rules()
            print(f"'{RULES_FILE_PATH}' already exists and is not empty. Skipping initial generation.")
    except Exception as e:
        print(f"Error during initial rule generation on startup: {e}")
    # --- END ADDITION ---

    start_background_thread() # This will now watch for *subsequent* changes
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))