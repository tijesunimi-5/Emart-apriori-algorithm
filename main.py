import os
import json
import asyncio
import logging
import random
import numpy as np
from pymongo import MongoClient
from apyori import apriori
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pymongo.errors import PyMongoError, ConnectionFailure, ServerSelectionTimeoutError
from pydantic import BaseModel
import uvicorn

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- FastAPI App Initialization ---
app = FastAPI()

# --- CORS Configuration ---
origins = [
    "https://e-mart-rho.vercel.app",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# --- Configuration with Environment Variables ---
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb+srv://tijesunimiidowu16:M7UN0QTHvX6P5ktw@cluster0.x5257.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./rules")
RULES_FILE_PATH = os.path.join(OUTPUT_DIR, "apriori_rules.json")

logger.info(f"Running on platform: {os.name}")
logger.info(f"Configured OUTPUT_DIR: {OUTPUT_DIR}")
logger.info(f"Expected RULES_FILE_PATH: {RULES_FILE_PATH}")

# Ensure output directory exists
try:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info(f"Ensured OUTPUT_DIR exists: {OUTPUT_DIR}")
except Exception as e:
    logger.error(f"Failed to create OUTPUT_DIR {OUTPUT_DIR}: {e}", exc_info=True)
    raise RuntimeError(f"Cannot initialize application: Failed to create output directory {OUTPUT_DIR}")

# --- MongoDB Connection Management ---
class MongoDBConnection:
    def __init__(self, uri):
        self.uri = uri
        self.client = None
        self._lock = asyncio.Lock()

    async def connect(self):
        async with self._lock:
            if not self.client or not self.client.is_connected:
                try:
                    self.client = MongoClient(
                        self.uri,
                        serverSelectionTimeoutMS=30000,
                        maxPoolSize=50,
                        connectTimeoutMS=20000,
                        socketTimeoutMS=20000
                    )
                    await asyncio.to_thread(self.client.admin.command, 'ping')
                    logger.info("Successfully connected to MongoDB.")
                except (PyMongoError, ConnectionFailure, ServerSelectionTimeoutError) as e:
                    logger.error(f"Failed to connect to MongoDB: {e}", exc_info=True)
                    raise
            return self

    def get_collection(self):
        if not self.client or not self.client.is_connected:
            raise ConnectionFailure("MongoDB connection is not available.")
        return self.client["ecommerce"]

    def close(self):
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed.")

# Initialize MongoDB connection
db_connection = MongoDBConnection(MONGODB_URI)

# --- Thompson Sampling Implementation ---
class ThompsonSampling:
    def __init__(self, db_connection):
        self.db_connection = db_connection

    def get_collection(self):
        """Get the Thompson Sampling statistics collection."""
        return self.db_connection.get_collection()["recommendation_stats"]

    def initialize_stats(self, rule):
        """Initialize stats for a new recommendation rule if not already present."""
        recommendation_id = f"{','.join(rule['antecedents'])}->{','.join(rule['consequents'])}"
        collection = self.get_collection()
        if not collection.find_one({"recommendation_id": recommendation_id}):
            collection.insert_one({
                "recommendation_id": recommendation_id,
                "antecedents": rule["antecedents"],
                "consequents": rule["consequents"],
                "successes": 0,
                "failures": 0
            })
            logger.info(f"Initialized Thompson Sampling stats for {recommendation_id}")

    def select_recommendation(self, rules):
        """Select a recommendation using Thompson Sampling."""
        if not rules:
            return None

        collection = self.get_collection()
        sampled_values = []
        for rule in rules:
            recommendation_id = f"{','.join(rule['antecedents'])}->{','.join(rule['consequents'])}"
            stats = collection.find_one({"recommendation_id": recommendation_id})
            if not stats:
                self.initialize_stats(rule)
                stats = {"successes": 0, "failures": 0}

            # Sample from Beta distribution
            sampled_value = np.random.beta(stats["successes"] + 1, stats["failures"] + 1)
            sampled_values.append((sampled_value, rule))

        # Select the rule with the highest sampled value
        if sampled_values:
            return max(sampled_values, key=lambda x: x[0])[1]
        return random.choice(rules)  # Fallback to random if no samples

    def update_stats(self, recommendation_id, success):
        """Update successes or failures for a recommendation."""
        collection = self.get_collection()
        try:
            if success:
                collection.update_one(
                    {"recommendation_id": recommendation_id},
                    {"$inc": {"successes": 1}}
                )
                logger.info(f"Incremented successes for {recommendation_id}")
            else:
                collection.update_one(
                    {"recommendation_id": recommendation_id},
                    {"$inc": {"failures": 1}}
                )
                logger.info(f"Incremented failures for {recommendation_id}")
        except PyMongoError as e:
            logger.error(f"Failed to update stats for {recommendation_id}: {e}", exc_info=True)

# --- Apriori Rule Generation ---
def fetch_transactions_sync():
    """Fetches transactions from MongoDB synchronously for Apriori."""
    try:
        with db_connection.connect() as conn:  # Ensure connection is active
            transactions_collection = conn.get_collection()["transactions"]
            transactions_data = transactions_collection.find({}, {"items": 1, "_id": 0})
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
        try:
            with open(RULES_FILE_PATH, "w") as file:
                json.dump([], file, indent=2)
            logger.info(f"Wrote empty rules to {RULES_FILE_PATH} as no transactions were found.")
        except Exception as e:
            logger.error(f"Failed to write empty rules file: {e}", exc_info=True)
        return

    try:
        rules_generator = apriori(
            transactions,
            min_support=0.001,
            min_confidence=0.2,
            min_lift=1.0,
            min_length=2
        )
        rules_list = []
        ts = ThompsonSampling(db_connection)  # Initialize Thompson Sampling
        for rule in rules_generator:
            for ordered_stat in rule.ordered_statistics:
                rule_data = {
                    "antecedents": list(ordered_stat.items_base),
                    "consequents": list(ordered_stat.items_add),
                    "support": rule.support,
                    "confidence": ordered_stat.confidence,
                    "lift": ordered_stat.lift
                }
                rules_list.append(rule_data)
                ts.initialize_stats(rule_data)  # Initialize stats for the rule
        with open(RULES_FILE_PATH, "w") as file:
            json.dump(rules_list, file, indent=2)
        logger.info(f"Updated {RULES_FILE_PATH} with {len(rules_list)} rules.")
    except Exception as e:
        logger.error(f"Error generating or saving Apriori rules: {e}", exc_info=True)

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
    await asyncio.to_thread(update_apriori_rules_sync)
    logger.info("Rule update process initiated (may take a moment to complete).")
    return {"message": "Rules update process initiated. Check logs for status."}

@app.post("/api/recommendationAPI")
async def get_recommendations(userItems: str = None):
    """Returns recommendations using Thompson Sampling based on user items."""
    logger.info(f"POST /api/recommendationAPI endpoint hit. Checking {RULES_FILE_PATH}")
    if not os.path.exists(RULES_FILE_PATH):
        logger.warning(f"RULES_FILE_PATH does NOT exist: {RULES_FILE_PATH}")
        raise HTTPException(status_code=404, detail="Rules not yet generated or file not found.")
    try:
        with open(RULES_FILE_PATH, "r") as file:
            rules_content = file.read()
            if not rules_content.strip():
                logger.warning(f"RULES_FILE_PATH is empty or only whitespace.")
                return {"recommendations": []}
            rules = json.loads(rules_content)
            logger.info(f"Successfully parsed {len(rules)} rules for recommendations.")
    except json.JSONDecodeError as e:
        logger.error(f"JSON Decode Error reading {RULES_FILE_PATH}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to parse rules file: Invalid JSON: {e}")
    except Exception as e:
        logger.error(f"Generic error in get_recommendations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

    user_items_list = json.loads(userItems) if userItems else []
    ts = ThompsonSampling(db_connection)

    if not user_items_list:  # Empty cart, select a recommendation
        if rules:
            selected_rule = ts.select_recommendation(rules)
            logger.info(f"Selected rule for empty cart: {selected_rule}")
            return {"recommendations": [selected_rule] if selected_rule else []}
        return {"recommendations": []}

    # Filter rules based on user items
    filtered_rules = [
        rule for rule in rules
        if set(user_items_list).issuperset(set(rule["antecedents"]))
    ]
    logger.info(f"Filtered {len(filtered_rules)} rules based on user items: {user_items_list}")

    # Select a recommendation using Thompson Sampling
    if filtered_rules:
        selected_rule = ts.select_recommendation(filtered_rules)
        return {"recommendations": [selected_rule] if selected_rule else []}
    return {"recommendations": []}

class FeedbackRequest(BaseModel):
    recommendation_id: str
    success: bool

@app.post("/api/recommendation-feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """Submits feedback for a recommendation to update Thompson Sampling stats."""
    logger.info(f"POST /api/recommendation-feedback for {feedback.recommendation_id}")
    ts = ThompsonSampling(db_connection)
    try:
        ts.update_stats(feedback.recommendation_id, feedback.success)
        return {"message": "Feedback recorded successfully"}
    except Exception as e:
        logger.error(f"Failed to record feedback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to record feedback: {str(e)}")

# --- Application Startup Function ---
async def start_application():
    """Initializes rule generation and starts the Uvicorn server."""
    logger.info("Starting application...")
    await db_connection.connect()  # Establish MongoDB connection
    # Create index for recommendation_stats collection
    try:
        db_connection.get_collection()["recommendation_stats"].create_index("recommendation_id")
        logger.info("Created index on recommendation_id for recommendation_stats collection")
    except PyMongoError as e:
        logger.error(f"Failed to create index on recommendation_stats: {e}", exc_info=True)
    await asyncio.to_thread(update_apriori_rules_sync)
    logger.info("Initial rule generation complete (if data available).")
    config = uvicorn.Config(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(start_application())