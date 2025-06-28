import sqlite3
from datetime import datetime
from pytz import timezone

IST = timezone('Asia/Kolkata')

# Function to create the page visited table
def create_page_visited_table():
    conn = sqlite3.connect('./data/data.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS page_visited_details
                 (pagename TEXT, time_of_visit TIMESTAMP)''')
    conn.commit()
    conn.close()

# Function to add page visited details
def add_page_visited_details(pagename, time_of_visit):
    conn = sqlite3.connect('./data/data.db')
    c = conn.cursor()
    c.execute('''INSERT INTO page_visited_details (pagename, time_of_visit)
                 VALUES (?, ?)''', (pagename, time_of_visit))
    conn.commit()
    conn.close()

# Function to view all page visited details
def view_all_page_visited_details():
    conn = sqlite3.connect('./data/data.db')
    c = conn.cursor()
    c.execute('''SELECT * FROM page_visited_details''')
    data = c.fetchall()
    conn.close()
    return data

# Function to create the emotion classifier table
def create_emotionclf_table():
    conn = sqlite3.connect('./data/data.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS emotionclf_table
                 (rawtext TEXT, prediction TEXT, probability REAL, time_of_visit TIMESTAMP)''')
    conn.commit()
    conn.close()

# Function to add prediction details
def add_prediction_details(rawtext, prediction, probability, time_of_visit):
    conn = sqlite3.connect('./data/data.db')
    c = conn.cursor()
    c.execute('''INSERT INTO emotionclf_table (rawtext, prediction, probability, time_of_visit)
                 VALUES (?, ?, ?, ?)''', (rawtext, prediction, probability, time_of_visit))
    conn.commit()
    conn.close()

# Function to view all prediction details
def view_all_prediction_details():
    conn = sqlite3.connect('./data/data.db')
    c = conn.cursor()
    c.execute('''SELECT * FROM emotionclf_table''')
    data = c.fetchall()
    conn.close()
    return data

# Function to create the health metrics table
def create_health_metrics_table():
    conn = sqlite3.connect('./data/data.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS health_metrics
                 (rawtext TEXT, temperature REAL, blood_pressure TEXT, time_of_visit TIMESTAMP)''')
    conn.commit()
    conn.close()

# Function to add health metrics
def add_health_metrics(rawtext, temperature, blood_pressure, time_of_visit):
    conn = sqlite3.connect('./data/data.db')
    c = conn.cursor()
    c.execute('''INSERT INTO health_metrics (rawtext, temperature, blood_pressure, time_of_visit)
                 VALUES (?, ?, ?, ?)''', (rawtext, temperature, blood_pressure, time_of_visit))
    conn.commit()
    conn.close()

# Function to view all health metrics
def view_all_health_metrics():
    conn = sqlite3.connect('./data/data.db')
    c = conn.cursor()
    c.execute('''SELECT * FROM health_metrics''')
    data = c.fetchall()
    conn.close()
    return data
    
    # Function to create the image emotion detection table
def create_image_emotion_table():
    conn = sqlite3.connect('./data/data.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS image_emotion_table
                 (image_path TEXT, emotion TEXT, time_of_detection TIMESTAMP)''')
    conn.commit()
    conn.close()

# Function to add image emotion detection details
def add_image_emotion_details(image_path, emotion, time_of_detection):
    conn = sqlite3.connect('./data/data.db')
    c = conn.cursor()
    c.execute('''INSERT INTO image_emotion_table (image_path, emotion, time_of_detection)
                 VALUES (?, ?, ?)''', (image_path, emotion, time_of_detection))
    conn.commit()
    conn.close()

# Function to view all image emotion detection details
def view_all_image_emotion_details():
    conn = sqlite3.connect('./data/data.db')
    c = conn.cursor()
    c.execute('''SELECT * FROM image_emotion_table''')
    data = c.fetchall()
    conn.close()
    return data

