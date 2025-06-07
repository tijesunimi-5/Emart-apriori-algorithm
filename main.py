import os
import json
import asyncio
import logging
from pymongo import MongoClient
from apyori import apriori # Make sure 'apyori' is in your requirements.txt
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pymongo.errors import PyMongoError
import uvicorn # Used for starting the server manually with server.serve()

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- FastAPI App Initialization ---
app = FastAPI()

# --- CORS Configuration ---
# IMPORTANT: Replace with your actual Vercel frontend URL when deployed.
# For local development, 'http://localhost:3000' is usually correct.
origins = [
    "https://e-mart-rho.vercel.app", # Your deployed Vercel frontend URL
    "http://localhost:3000",        # Your local frontend URL
    # "*" # Avoid using "*" in production for security reasons
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"], # Explicitly allow methods
    allow_headers=["*"],
)

# --- Configuration with Environment Variables ---
# MongoDB URI: Get from environment variable or use default
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb+srv://tijesunimiidowu16:M7UN0QTHvX6P5ktw@cluster0.x5257.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")

# OUTPUT_DIR: Where to store generated Apriori rules.
# On Render, /opt/render/project/src/ is the root of your cloned repository.
# './rules' creates a 'rules' directory inside your project's root.
# This ensures it's written to a writable location.
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./rules") # Relative path within the project

RULES_FILE_PATH = os.path.join(OUTPUT_DIR, "apriori_rules.json")

logger.info(f"Running on platform: {os.name}")
logger.info(f"Configured OUTPUT_DIR: {OUTPUT_DIR}")
logger.info(f"Expected RULES_FILE_PATH: {RULES_FILE_PATH}")

# Ensure output directory exists when the app starts
# This will create `./rules` if it doesn't exist within the cloned repo.
try:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info(f"Ensured OUTPUT_DIR exists: {OUTPUT_DIR}")
except Exception as e:
    logger.error(f"Failed to create OUTPUT_DIR {OUTPUT_DIR}: {e}", exc_info=True)
    # If this fails, the app cannot proceed, so re-raise or handle gracefully
    raise RuntimeError(f"Cannot initialize application: Failed to create output directory {OUTPUT_DIR}")

# --- MongoDB Connection ---
mongo_client = None # Global variable to hold client instance
transactions_collection = None

@app.on_event("startup")
async def connect_to_mongodb():
    """Connects to MongoDB and initializes collections on app startup."""
    global mongo_client, transactions_collection
    try:
        mongo_client = MongoClient(MONGODB_URI)
        db = mongo_client["ecommerce"]
        transactions_collection = db["transactions"]
        # The ping command is cheap and high-performance.
        # It ensures that the driver can connect to the database.
        mongo_client.admin.command('ping')
        logger.info(f"Successfully connected to MongoDB. Using DB: {db.name}")
    except PyMongoError as e:
        logger.error(f"Failed to connect to MongoDB: {e}", exc_info=True)
        # Depending on criticality, you might want to raise an exception here
        # to prevent the app from starting if the DB connection is essential.
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during MongoDB connection: {e}", exc_info=True)
        raise

@app.on_event("shutdown")
async def close_mongodb_connection():
    """Closes the MongoDB connection on app shutdown."""
    if mongo_client:
        mongo_client.close()
        logger.info("MongoDB connection closed.")

# --- Apriori Rule Generation ---
def fetch_transactions_sync():
    """Fetches transactions from MongoDB synchronously for Apriori."""
    try:
        # Use transactions_collection.find() which returns a cursor
        # Ensure it's not trying to call this before connection is established
        if transactions_collection is None:
            logger.error("Transactions collection not initialized.")
            return []
            
        transactions_data = transactions_collection.find({}, {"items": 1, "_id": 0})
        # Extract product IDs. Ensure 'productId' is consistently named in your transaction items.
        transactions = [[item.get("productId") for item in tx.get("items", []) if item.get("productId")]
                        for tx in transactions_data if tx.get("items")]
        logger.info(f"Fetched {len(transactions)} transactions for Apriori.")
        return transactions
    except PyMongoError as e:
        logger.error(f"Failed to fetch transactions for Apriori: {e}", exc_info=True)
        return []
    except Exception as e:
        logger.error(f"Unexpected error in fetch_transactions_sync: {e}", exc_info=True)
        return []

def update_apriori_rules_sync():
    """Updates Apriori rules and saves them to a JSON file synchronously."""
    transactions = fetch_transactions_sync()
    if not transactions:
        logger.warning("No transactions fetched for Apriori. Skipping rule generation.")
        # Ensure the rules file is empty or updated to reflect no rules
        try:
            with open(RULES_FILE_PATH, "w") as file:
                json.dump([], file, indent=2) # Write an empty array if no rules
            logger.info(f"Wrote empty rules to {RULES_FILE_PATH} as no transactions were found.")
        except Exception as e:
            logger.error(f"Failed to write empty rules file: {e}", exc_info=True)
        return

    try:
        # apyori expects a generator or iterable of iterables
        rules_generator = apriori(
            transactions,
            min_support=0.001, # Adjust these parameters based on your data and desired rule count
            min_confidence=0.2,
            min_lift=1.0,
            min_length=2 # Minimum number of items in a rule (antecedent + consequent)
        )

        rules_list = []
        for rule in rules_generator:
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
        logger.info(f"Updated {RULES_FILE_PATH} with {len(rules_list)} rules.")
    except Exception as e:
        logger.error(f"Error generating or saving Apriori rules: {e}", exc_info=True)
        # Don't re-raise, allow the app to start even if rules generation fails initially

# --- API Endpoints ---
@app.get("/api/rules")
async def get_rules():
    """Returns the generated Apriori rules."""
    logger.info(f"GET /api/rules endpoint hit. Checking {RULES_FILE_PATH}")
    if not os.path.exists(RULES_FILE_PATH):
        logger.warning(f"RULES_FILE_PATH does NOT exist: {RULES_FILE_PATH}")
        raise HTTPException(status_code=404, detail="Rules not yet generated or file not found.")

    try:
        with open(RULES_FILE_PATH, "r") as file:
            rules_content = file.read()
            if not rules_content.strip():
                logger.warning(f"RULES_FILE_PATH is empty or only whitespace.")
                # Return empty list if file is empty but exists, indicating no rules
                return []
            rules = json.loads(rules_content)
            logger.info(f"Successfully parsed {len(rules)} rules.")
            return rules
    except json.JSONDecodeError as e:
        logger.error(f"JSON Decode Error reading {RULES_FILE_PATH}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to parse rules file: Invalid JSON: {e}")
    except Exception as e:
        logger.error(f"Generic error in get_rules: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.post("/api/update-rules")
async def update_rules_endpoint():
    """Triggers an update of the Apriori rules."""
    logger.info("POST /api/update-rules endpoint hit. Triggering rule update...")
    # Run the synchronous function in an executor to avoid blocking the event loop
    await asyncio.to_thread(update_apriori_rules_sync)
    logger.info("Rule update process initiated (may take a moment to complete).")
    return {"message": "Rules update process initiated. Check logs for status."}

# --- Application Startup Function ---
async def start_application():
    """Initializes rule generation and starts the Uvicorn server."""
    logger.info("Starting application...")
    # Force initial rule generation on startup
    # Run synchronously in a separate thread to avoid blocking startup
    await asyncio.to_thread(update_apriori_rules_sync)
    logger.info("Initial rule generation complete (if data available).")

    # Uvicorn server setup (manual run)
    config = uvicorn.Config(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
    server = uvicorn.Server(config)
    await server.serve()

# --- Entry Point ---
if __name__ == "__main__":
    # Use asyncio.run to start the main async function
    asyncio.run(start_application())

