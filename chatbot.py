import os
import logging
import mimetypes
import asyncio
from datetime import datetime
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    CommandHandler,
    MessageHandler,
    filters
)
from pymongo import MongoClient
import google.generativeai as genai

# Environment setup
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")     
MONGO_URI = os.getenv("MONGO_URI")                 
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") 
# Database initialization with connection pooling
mongo_client = MongoClient(MONGO_URI, maxPoolSize=50, connectTimeoutMS=30000)
db = mongo_client.telegram_bot_db

# Create indexes for faster queries
db.users.create_index("chat_id", unique=True)
db.chat_history.create_index([("user_id", 1), ("timestamp", -1)])
db.files.create_index("user_id")
db.searches.create_index([("user_id", 1), ("timestamp", -1)])

# AI configuration
genai.configure(api_key=GEMINI_API_KEY)
generation_config = {
    "temperature": 0.9,
    "top_p": 1,
    "max_output_tokens": 2048,
}

nlp_model = genai.GenerativeModel('gemini-pro', generation_config=generation_config)
vision_model = genai.GenerativeModel('gemini-1.5-flash', generation_config=generation_config)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache for frequently accessed data
user_cache = {}

# === Fast Response Times ===
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle initial registration with state tracking."""
    user = update.effective_user
    users = db.users
    
    # Check cache first
    if user.id in user_cache:
        await update.message.reply_text(f"Welcome back {user.first_name}! Your registration is complete.")
        return

    existing_user = users.find_one({"chat_id": user.id})
    if not existing_user:
        users.insert_one({
            "first_name": user.first_name,
            "username": user.username,
            "chat_id": user.id,
            "status": "pending_contact",
            "registered_at": datetime.now(),
            "last_interaction": datetime.now()
        })
        await request_contact(update, context)
    else:
        status = existing_user.get("status", "new")
        if status == "pending_contact":
            await update.message.reply_text("Please complete registration by sharing your contact")
        else:
            user_cache[user.id] = existing_user  # Cache the user
            await update.message.reply_text(f"Welcome back {user.first_name}! Your registration is complete.")

async def request_contact(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send contact request inline keyboard."""
    keyboard = [[InlineKeyboardButton("Verify Phone Number", request_contact=True)]]
    await update.message.reply_text(
        "📱 Verification Required\nTo use this bot, please share your phone number for security:",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def handle_contact(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Process received contact information."""
    contact = update.effective_message.contact
    users = db.users
    
    logger.info(f"Received contact: {contact}")
    
    # Security validation
    if contact.user_id != update.effective_user.id:
        await update.message.reply_text("⚠️ Please only share your own contact information.")
        return

    update_result = users.update_one(
        {"chat_id": contact.user_id},
        {"$set": {
            "phone_number": contact.phone_number,
            "status": "verified",
            "phone_verified_at": datetime.now()
        }}
    )
    
    if update_result.modified_count == 1:
        user_cache[contact.user_id] = users.find_one({"chat_id": contact.user_id})  # Cache the user
        await update.message.reply_text("✅ Phone verification successful!\nYou can now access all bot features.")
    else:
        await update.message.reply_text("❌ Verification failed. Please try /start again")

# === Minimal API Latency ===
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle text messages and reply with Gemini NLP."""
    user_id = update.effective_user.id
    user_message = update.message.text

    # Offload Gemini API call to a background task
    asyncio.create_task(generate_and_send_response(update, user_id, user_message))

async def generate_and_send_response(update: Update, user_id: int, user_message: str):
    """Generate a response using Gemini and send it."""
    try:
        # Add emoji to the prompt for creative responses
        prompt = f"Respond to this message with a helpful and friendly tone, and include relevant emojis: {user_message}"
        raw_response = await asyncio.to_thread(nlp_model.generate_content, prompt)
        gemini_response = extract_text_from_candidates(raw_response)

        # Save chat history
        db.chat_history.insert_one({
            "user_id": user_id,
            "input": user_message,
            "response": gemini_response,
            "timestamp": datetime.now()
        })

        await update.message.reply_text(gemini_response)

        # Auto-follow-up after 5 seconds
        await asyncio.sleep(5)
        follow_up = "Is there anything else I can help you with? 😊"
        await update.message.reply_text(follow_up)
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        await update.message.reply_text("❌ Sorry, I couldn't process your request. Please try again.")

# === Efficient DB Storage ===
async def handle_files(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Process image/file uploads using Gemini vision."""
    try:
        if update.message.photo:
            file_id = update.message.photo[-1].file_id
        elif update.message.document:
            file_id = update.message.document.file_id
        else:
            await update.message.reply_text("❌ Unsupported file type")
            return

        # Download the file
        file = await context.bot.get_file(file_id)
        file_path = await file.download_to_drive()
        
        # Read file content
        with open(file_path, 'rb') as f:
            file_data = f.read()

        # Get MIME type
        mime_type, _ = mimetypes.guess_type(file_path)
        mime_type = mime_type or 'application/octet-stream'

        # Generate analysis
        prompt = "Analyze and describe this content in detail:"
        response = await asyncio.to_thread(vision_model.generate_content, [prompt, {"mime_type": mime_type, "data": file_data}])
        analysis = extract_text_from_candidates(response)

        # Save to DB and reply
        db.files.insert_one({
            "user_id": update.effective_user.id,
            "filename": os.path.basename(file_path),
            "file_type": mime_type.split('/')[-1],
            "analysis": analysis,
            "timestamp": datetime.now()
        })
        await update.message.reply_text(f"🔍 Analysis Result:\n{analysis}")
    except Exception as e:
        logger.error(f"File processing error: {e}")
        await update.message.reply_text("❌ Failed to analyze file")

def extract_text_from_candidates(response):
    """Extract text from the first candidate, concatenating all parts if multiple parts exist."""
    if not response.candidates:
        return "No candidates returned."

    text_parts = []
    for part in response.candidates[0].content.parts:
        text_parts.append(part.text)
    return "".join(text_parts)

# === Unique Feature: Context-Aware Responses ===
async def handle_websearch(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Perform a simple 'AI-based' web search query."""
    query = " ".join(context.args)
    if not query:
        await update.message.reply_text("Usage: /websearch <your query here>")
        return

    # Summarize search with Gemini
    web_search_prompt = f"Simulate a web search for: {query} and provide a concise summary with top links."
    raw_response = await asyncio.to_thread(nlp_model.generate_content, web_search_prompt)
    gemini_response = extract_text_from_candidates(raw_response)

    db.searches.insert_one({
        "user_id": update.effective_user.id,
        "query": query,
        "result": gemini_response,
        "timestamp": datetime.now()
    })

    await update.message.reply_text(f"🌐 Search Results:\n{gemini_response}")

def main():
    """Run the Telegram bot application."""
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    # Handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("websearch", handle_websearch))
    application.add_handler(MessageHandler(filters.CONTACT, handle_contact))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(MessageHandler(filters.PHOTO | filters.Document.ALL, handle_files))

    application.run_polling()

if __name__ == "__main__":
    main()
