import streamlit as st
import sqlite3
import torch
import random
import json
from passlib.hash import pbkdf2_sha256
import google.generativeai as genai
from nltk_utils import tokenize, bag_of_words
from model import NeuralNet
import csv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

genai.configure(api_key="AIzaSyA88NRkMNFZVpVuZGpdofWx49qor8O80ew")
generation_config = {"temperature": 0.9, "top_p": 1, "top_k": 1, "max_output_tokens": 2048}

model2 = genai.GenerativeModel("gemini-pro", generation_config=generation_config)
model1 = genai.GenerativeModel("gemini-pro-vision", generation_config=generation_config)
FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"

# Function to create a connection to the database
def create_connection():
    return sqlite3.connect('users.db')

# Function to create the users table if it doesn't exist
def create_table():
    conn = create_connection()
    with conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                password TEXT NOT NULL
            )
        ''')

# Function to register a new user
def register_user(username, password):
    conn = create_connection()
    with conn:
        hashed_password = pbkdf2_sha256.hash(password)
        conn.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed_password))

# Function to authenticate a user
def authenticate_user(username, password):
    conn = create_connection()
    with conn:
        cursor = conn.execute('SELECT password FROM users WHERE username = ?', (username,))
        result = cursor.fetchone()
        if result:
            return pbkdf2_sha256.verify(password, result[0])
        return False

# Function to get response from the chatbot
def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    
    return "I do not understand..."

# Main part of the Streamlit app
def main():
    st.title('WELCOME TO GEMINI')

    # Create the users table if it doesn't exist
    create_table()

    if 'page' not in st.session_state:
        st.session_state.page = "home"
        st.session_state.logged_in = False
        st.session_state.history = []

    if st.session_state.page == "home":
        st.subheader('Home')
        st.markdown("CourseBot is your personalized learning assistant!")

        if st.button("Login"):
            st.session_state.page = "login"
        if st.button("Sign Up"):
            st.session_state.page = "register"

    elif st.session_state.page == 'login':
        st.subheader('Login')
        username = st.text_input('Username')
        password = st.text_input('Password', type='password')

        if st.button('Login'):
            if authenticate_user(username, password):
                st.success(f'Welcome, {username}!')
                st.session_state.logged_in = True
                st.session_state.page = "chat"
            else:
                st.error('Authentication failed. Please check your username and password.')

    elif st.session_state.page == 'register':
        st.subheader('Register')
        new_username = st.text_input('New Username')
        new_password = st.text_input('New Password', type='password')

        if st.button('Register'):
            register_user(new_username, new_password)
            st.success(f'Account created for {new_username}. You can now log in.')
            st.session_state.page = "login"

    elif st.session_state.page == 'chat':
        st.title("EDUGENIE🧠")
        user_input = st.text_input("You: ")

        if st.button("Send"):
            response = get_response(user_input)
            st.text(response)
            st.markdown(
                f'<div style="background-color:#2C0A7C; padding:10px; border-radius:10px;">{response}</div>',
                unsafe_allow_html=True
            )
            st.session_state.history.append((user_input, response))
            with open('chat_history.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([user_input, response])
            st.subheader("Chat History")
            for user_input, response in st.session_state.history:
                st.text(f"You: {user_input}")
                st.text(f"EDUGENIE🧠: {response}")
                st.text("")
        # Button for Doubt Assistant
        # Button for Doubt Assistant
        if st.button("Doubt Assistant", key="doubt_assistant_button"):
            st.session_state.page = "doubt_assistant"

        # In the main part of your Streamlit app
    elif st.session_state.page == "doubt_assistant":
            st.title("Gemini Pro Chat Room")
            user_input = st.text_input("You: ")

            if st.button("Send"):
                response = model2.generate_content([user_input])
                for chunk in response:
                    st.text("Gemini Pro: " + chunk.text)
                    st.markdown(
                    f'<div style="background-color:#2C0A7C; padding:10px; border-radius:10px;">{response}</div>',
                    unsafe_allow_html=True
                )
                st.session_state.history.append((user_input, response))
                response_text = ' '.join([chunk.text for chunk in response])

# Write the user input and the concatenated response to the CSV file
                with open('doubt_chat_history.csv', mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([user_input, response_text])

                st.subheader("Chat History")
                for user_input, response in st.session_state.history:
                    st.text(f"You: {user_input}")
                    for chunk in response:
                        st.text("Gemini Pro: "+chunk.text)
                    st.text("")

        

if __name__ == '__main__':
    main()
