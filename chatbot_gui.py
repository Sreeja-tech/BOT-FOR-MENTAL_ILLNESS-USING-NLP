
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Function to preprocess text
def preprocess_text(text):
    text = text.lower() # Lower
    text = re.sub(r'[^a-z\s]', '', text) # Remove punctuation and non-alphabetic characters
    tokens = word_tokenize(text) # Tokenise
    stop_words = set(stopwords.words('english')) # Remove stop words
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer() # Lemmatize text
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)  # Rejoin tokens into a single string

# Function to find the most similar question and retrieve the corresponding answer
def get_response(user_input, data, vectorizer, tfidf_matrix):
    # Preprocess the user's input
    preprocessed_input = preprocess_text(user_input)
    # Vectorize the user's input using the already-fitted vectorizer
    user_input_vector = vectorizer.transform([preprocessed_input])

    # Compute the cosine similarity between the user's input and all preprocessed questions
    similarities = cosine_similarity(user_input_vector, tfidf_matrix)

    # Find the maximum similarity score and its index
    max_similarity_score = max(similarities[0])
    most_similar_question_index = similarities.argmax()

    # If the maximum similarity score is below the threshold, return a message indicating the limitation
    if max_similarity_score < 0.5:
        return None, "I am sorry, but I need to learn more to answer that question properly."
        

    # Find the index of the most similar question
    most_similar_question_index = similarities.argmax()

    # Retrieve the most similar question and its answer
    similar_question = data.iloc[most_similar_question_index]['Questions']
    response = data.iloc[most_similar_question_index]['Answers']

    return similar_question, response

# Load the data
# NOTE: The data should be in the same directory as this script or provide the full path.
file_path = 'Mental_Health_FAQ.csv'  # Replace with the correct path to the CSV file
data = pd.read_csv(file_path)
data['Processed_Question'] = data['Questions'].apply(preprocess_text)

# Create the TF-IDF vectorizer and vectorize the preprocessed questions
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(data['Processed_Question'])

# GUI Application
class ChatbotApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Mental Health FAQ Chatbot')
        self.geometry('600x400')  # Set a default size for the window
        
        style = ttk.Style(self)
        style.configure('TButton', font=('Helvetica', 12), padding=10)
        style.configure('TLabel', font=('Helvetica', 12))
        style.configure('TEntry', font=('Helvetica', 12), padding=10)

        # User input text field
        self.user_input_label = ttk.Label(self, text='Hello, I am here to help. Ask a question related to mental health:')
        self.user_input_label.pack(padx=20, pady=(20, 0))
        self.user_input = ttk.Entry(self, width=50)
        self.user_input.pack(padx=20, pady=10, fill='x', expand=True)

        # Submit button
        self.submit_button = ttk.Button(self, text='Ask', command=self.get_bot_response)
        self.submit_button.pack(pady=10)

        # Chat display area
        self.chat_display = ScrolledText(self, state='disabled', width=75, height=20)
        self.chat_display.pack(padx=20, pady=10, fill='both', expand=True)
    
    def get_bot_response(self):
        # Get the user input
        user_question = self.user_input.get()
        # Clear the input field
        self.user_input.delete(0, tk.END)
        # Get the response from the chatbot
        similar_question, answer = get_response(user_question, data, vectorizer, tfidf_matrix)
        # Display the response
        self.chat_display.config(state='normal')
        self.chat_display.insert(tk.END, "You: " + user_question + "\n")
        self.chat_display.insert(tk.END, "Bot: " + answer + "\n\n")
        self.chat_display.config(state='disabled')
        self.chat_display.yview(tk.END)

# Run the application
if __name__ == "__main__":
    app = ChatbotApp()
    app.mainloop()
