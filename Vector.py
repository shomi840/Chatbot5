from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Define the FAQ dictionary (knowledge base)
faq_data = {
    "Do you offer delivery?": "Yes, we provide delivery service for all orders above 500 BDT.",
    "Where are you located?": "We are located at 123 Main Road, Dhaka.",
    "Can I return a product?": "You can return a product within 7 days of delivery.",
    "What payment methods do you accept?": "We accept cash on delivery, bKash, and credit card payments.",
    "How long does delivery take?": "Delivery usually takes 2-3 business days."
}

# Step 2: Prepare the vectorizer and fit it on the FAQ questions
questions = list(faq_data.keys())
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)

# Step 3: Define the function to get the best matching answer
def get_faq_answer(user_query: str) -> str:
    # Vectorize the user query using the same vectorizer
    user_vector = vectorizer.transform([user_query])
    
    # Compute cosine similarity between user query and all FAQ questions
    similarities = cosine_similarity(user_vector, question_vectors)
    
    # Get the index and score of the most similar question
    max_index = similarities.argmax()
    max_score = similarities[0, max_index]

    # Step 4: Return answer if similarity is above threshold
    if max_score > 0.5:
        best_question = questions[max_index]
        return faq_data[best_question]
    else:
        return "Sorry, I couldn’t find a suitable answer to your question."

# Step 5: Example test cases
if __name__ == "__main__":
    print(get_faq_answer("Do you offer delivery?"))               # Should match
    print(get_faq_answer("Where are you located?"))               # Should match
    print(get_faq_answer("Can I return a product?"))              # Should match
    print(get_faq_answer("Do you sell mobile phones?"))           # Should not match
    print(get_faq_answer("Is cash on delivery available?"))       # Should match payment methods
