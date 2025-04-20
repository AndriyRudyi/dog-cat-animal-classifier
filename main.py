import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions
import numpy as np
import os
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
import logging
from datetime import datetime

#  Logging settings
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler("logs.txt"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


model = EfficientNetB0(weights='imagenet')

#  Keywords for identifying dogs and cats
dog_keywords = ["dog", "affenpinscher", "afghan hound", "african hunting dog", "airedale", "american staffordshire terrier",
    "appenzeller", "australian terrier", "basenji", "basset", "beagle", "bedlington terrier", "bernese mountain dog",
    "black and tan coonhound", "blenheim spaniel", "bloodhound", "bluetick", "border collie", "border terrier",
    "borzoi", "boston bull", "bouvier des flandres", "boxer", "brabancon griffon", "briard", "brittany spaniel",
    "bull mastiff", "cairn", "cardigan", "chesapeake bay retriever", "chihuahua", "chow", "clumber", "cocker spaniel",
    "collie", "curly-coated retriever", "dachshund", "dalmatian", "dandie dinmont", "doberman", "english foxhound",
    "english setter", "english springer", "entlebucher", "eskimo dog", "flat-coated retriever", "french bulldog",
    "german shepherd", "german short-haired pointer", "giant schnauzer", "golden retriever", "gordon setter",
    "great dane", "great pyrenees", "greater swiss mountain dog", "groenendael", "ibizan hound", "irish setter",
    "irish terrier", "irish water spaniel", "irish wolfhound", "italian greyhound", "japanese spaniel", "keeshond",
    "kelpie", "kerry blue terrier", "komondor", "kuvasz", "labrador retriever", "lakeland terrier", "leonberg",
    "lhasa", "malamute", "malinois", "maltese dog", "mexican hairless", "miniature pinscher", "miniature poodle",
    "miniature schnauzer", "newfoundland", "norfolk terrier", "norwegian elkhound", "norwich terrier", "old english sheepdog",
    "otterhound", "papillon", "pekinese", "pembroke", "petit basset griffon vendeen", "pharaoh hound", "plott",
    "pointer", "pomeranian", "pug", "redbone", "rhodesian ridgeback", "rottweiler", "saint bernard", "saluki",
    "samoyed", "schipperke", "scotch terrier", "scottish deerhound", "sealyham terrier", "shetland sheepdog",
    "shih-tzu", "siberian husky", "silky terrier", "soft-coated wheaten terrier", "staffordshire bullterrier",
    "standard poodle", "standard schnauzer", "sussex spaniel", "tibetan mastiff", "tibetan terrier", "toy poodle",
    "toy terrier", "vizsla", "walker hound", "weimaraner", "welsh springer spaniel", "west highland white terrier",
    "whippet", "wire-haired fox terrier", "yorkshire terrier"
]
cat_keywords = ["cat", "abyssinian", "american bobtail", "american curl", "american shorthair", "american wirehair", "balinese",
    "bengal", "birman", "bombay", "british shorthair", "burmese", "chartreux", "chausie", "cornish rex", "devon rex",
    "egyptian mau", "exotic shorthair", "havana brown", "himalayan", "japanese bobtail", "javanese", "korat",
    "laPerm", "maine coon", "manx", "norwegian forest", "ocicat", "oriental shorthair", "persian", "ragdoll",
    "russian blue", "savannah", "scottish fold", "selkirk rex", "siamese", "siberian", "singapura", "somali",
    "sphynx", "tonkinese", "turkish angora", "turkish van"
]

#


def predict_image(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        preds = model.predict(x)
        label = decode_predictions(preds, top=1)[0][0][1].replace('_', ' ').lower()

        # Keyword check
        if any(keyword in label for keyword in cat_keywords):
            return f"It's a cat! Breed: {label.capitalize()}"
        elif any(keyword in label for keyword in dog_keywords):
            return f"It's a dog! Breed: {label.capitalize()}"
        else:
            return f"Could not determine. Identified as: {label.capitalize()}"
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return f"An error occurred while processing the image: {str(e)}"

#  Saving query history
def save_history(user, message, result):
    with open("logs.txt", "a", encoding="utf-8") as file:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file.write(f"[{timestamp}] User: {user}, Message: {message}, Result: {result}\n")


TOKEN = 'tokenkoken'

#  /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hi, send me a photo of a cat or dog, and I'll tell you who it is and what breed. Test AI model by Andrii.R.")


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        user = update.effective_user.username or "unknown_user"
        photo = update.message.photo[-1]
        file = await photo.get_file()
        file_path = "received_image.jpg"
        await file.download_to_drive(file_path)

        result = predict_image(file_path)
        await update.message.reply_text(result)

        save_history(user, "Photo received", result)
        logger.info(f"Photo processed from {user}: {result}")

        os.remove(file_path)
    except Exception as e:
        logger.error(f"Photo processing error: {e}")
        await update.message.reply_text(f"An error occurred: {str(e)}")


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user.username or "unknown_user"
    message = update.message.text
    result = "Please send only the photo!"

    await update.message.reply_text(result)
    save_history(user, message, result)
    logger.warning(f"Unsupported message received from {user}: {message}")


async def handle_other(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user.username or "unknown_user"
    result = "Please send only the photo!"
    await update.message.reply_text(result)
    save_history(user, "Unsupported message type", result)
    logger.warning(f"Unsupported message type received from {user}")

if __name__ == "__main__":
    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT, handle_text))
    app.add_handler(MessageHandler(~filters.PHOTO & ~filters.TEXT, handle_other))

    print("✅ Bot started...")
    app.run_polling()
