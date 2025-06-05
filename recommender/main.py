from semantic_book_recommender import SemanticBookRecommender

def main():
    recommender = SemanticBookRecommender()
    recommender.load_and_preprocess_data("dataset_gemini_2_0_flash.csv")

    test_queries = [
        "science fiction with rating > 4.0",
        "by George Orwell",
        "mystery books with emotion sadness",
        "books with rating > 4.5",
        "fantasy by J.K. Rowling",
        "romance books"
    ]

    for query in test_queries:
        print(f"\n=== Query: \"{query}\" ===")
        results = recommender.find_similar_books(query, top_k=3)
        if results.empty:
            print("No matching books found.")
        else:
            for idx, row in results.iterrows():
                print(f"- {row['Title']} by {row['Author']} | Rating: {row['Rating']} | Emotion: {row['Emotion']} | Genres: {row['Genres']}")

if __name__ == "__main__":
    main()
