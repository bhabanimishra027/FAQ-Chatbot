import tkinter as tk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
nltk.download('punkt')

faq_data = {
    "What is AI?": "AI stands for Artificial Intelligence.",
    "What is Machine Learning?": "Machine Learning is a subset of AI that allows systems to learn from data.",
    "What is NLP?": "NLP stands for Natural Language Processing.",
    "What is deep learning?": "Deep learning is a type of machine learning using neural networks.",
    "What is a neural network?": "A neural network is a series of algorithms that mimic the human brain.",
    "What is supervised learning?": "Supervised learning is a type of ML where models are trained on labeled data.",
    "What is unsupervised learning?": "Unsupervised learning is a type of ML that works on unlabeled data.",
    "What is reinforcement learning?": "Reinforcement learning is based on rewarding or penalizing an agent.",
    "What is computer vision?": "Computer vision enables machines to interpret and understand images.",
    "What is a chatbot?": "A chatbot is a program that simulates conversation with users.",
    "What is an algorithm?": "An algorithm is a step-by-step procedure to solve a problem.",
    "What is a dataset?": "A dataset is a collection of data used for training or testing models.",
    "What is classification in ML?": "Classification is predicting a category label for given data.",
    "What is regression in ML?": "Regression is predicting a continuous value from data.",
    "What is overfitting in ML?": "Overfitting means the model performs well on training data but poorly on unseen data."
}

faq_questions = list(faq_data.keys())
faq_answers = list(faq_data.values())

def get_best_match(user_input):
    all_questions = faq_questions + [user_input]
    vectorizer = TfidfVectorizer().fit_transform(all_questions)
    similarity = cosine_similarity(vectorizer[-1], vectorizer[:-1])
    index = similarity.argmax()
    return faq_answers[index]

def get_response():
    user_input = user_entry.get()
    if user_input.strip() == "":
        return
    response = get_best_match(user_input)
    chat_log.insert(tk.END, "You: " + user_input + "\n")
    chat_log.insert(tk.END, "Bot: " + response + "\n\n")
    user_entry.delete(0, tk.END)


root = tk.Tk()
root.title("FAQ Chatbot")
root.geometry("500x400")

chat_log = tk.Text(root, height=20, width=60)
chat_log.pack(pady=10)

user_entry = tk.Entry(root, width=40)
user_entry.pack(side=tk.LEFT, padx=(10, 0), pady=10)

send_button = tk.Button(root, text="Send", command=get_response)
send_button.pack(side=tk.LEFT, padx=10)

root.mainloop()