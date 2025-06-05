import gradio as gr
import pandas as pd
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(project_root)
import asyncio
from semantic_book_recommender import SemanticBookRecommender
from object_detection.YoloModel import YoloModel
from ocr.OCRModels.GoogleVisionOCR import GoogleVisionOCR
from object_detection.ResultsProcessor import ResultsProcessor
from ocr_parser.BookOCRProcessor import BookOCRProcessor

recommender = SemanticBookRecommender()
# Vector DB is only created after image upload and processing.

def process_image(image_path):
    #os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "../ocr/hybrid-formula-458421-k0-e569816e1234.json"
    model = YoloModel("object_detection/obb/v8x/yolov8x-obb-best_20240202.pt")
    ocr_model = GoogleVisionOCR()
    results = model.predict(image_path, save=False, show=False)
    results_processor = ResultsProcessor()
    cropped_images = results_processor.extract_book_spines(image_path, results)

    texts = []
    for cropped_image in cropped_images:
        if cropped_image is not None:
            detected_text = ocr_model.detect_text_from_array(cropped_image)
            texts.append(detected_text)

    processor = BookOCRProcessor()

    async def run_pipeline():
        await processor.process(texts, "processed_books_dataset.csv")
        recommender.load_and_preprocess_data("processed_books_dataset.csv")

    asyncio.run(run_pipeline())
    return "Image processed and books updated."

def search_books(query, genre, emotion, min_rating, max_rating, top_k):
    result = recommender.find_similar_books(
        query=query,
        top_k=top_k,
        genre_filter=None if genre == "All" else genre,
        emotion_filter=None if emotion == "All" else emotion
    )

    if min_rating is not None:
        result = result[result['Rating'] >= min_rating]
    if max_rating is not None:
        result = result[result['Rating'] <= max_rating]

    gallery_data = []
    for _, row in result.iterrows():
        image_url = row['imageLink'] if pd.notna(row['imageLink']) and row['imageLink'].startswith("http") else "cover-not-found.jpg"
        title = row['Title']
        author = row['Author']
        rating = row['Rating']
        genres = row['Genres']
        emotion = row['Emotion']
        description = row['Description'] if pd.notna(row['Description']) else "No description available."

        caption = f"{title} by {author}\nRating: {rating}\nGenres: {genres}\nEmotion: {emotion}\n\n{description}"
        gallery_data.append([image_url, caption])

    return gallery_data

with gr.Blocks() as demo:
    gr.Markdown("## 📚 Semantic Book Recommender")

    with gr.Row():
        image_input = gr.Image(type="filepath", label="Upload Bookshelf Image")
        process_btn = gr.Button("Process Image")
        image_status = gr.Textbox(label="Status", interactive=False)

    process_btn.click(fn=process_image, inputs=image_input, outputs=image_status)

    with gr.Row():
        query = gr.Textbox(label="Search Query", placeholder="e.g. fantasy books by Tolkien")

    with gr.Row():
        genre = gr.Dropdown(choices=["All"], value="All", label="Genre Filter")
        emotion = gr.Dropdown(choices=["All"], value="All", label="Emotion Filter")

    with gr.Row():
        min_rating = gr.Slider(minimum=0.0, maximum=5.0, value=0.0, step=0.1, label="Min Rating")
        max_rating = gr.Slider(minimum=0.0, maximum=5.0, value=5.0, step=0.1, label="Max Rating")

    top_k = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Number of Results")
    search_btn = gr.Button("Search")

    result_gallery = gr.Gallery(label="Search Results", show_label=False, columns=5, height="auto")

    search_btn.click(fn=search_books, inputs=[query, genre, emotion, min_rating, max_rating, top_k], outputs=result_gallery)

demo.launch()
