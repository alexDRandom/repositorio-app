from fastapi import FastAPI, Request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import json
from jinja2 import Environment, FileSystemLoader
from fastapi.responses import HTMLResponse
from scipy.sparse import vstack
from sklearn.svm import SVR
import jsonlines
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

app = FastAPI()
env = Environment(loader=FileSystemLoader("templates"))
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Ruta para la página de inicio con el formulario
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})



import json

def get_reviews_by_gmap_ids(gmap_id):
    review_low = []
    review_high = []
    total_rating = 0
    count = 0
    
    with jsonlines.open('data/strict-filtered-reviews.jsonl') as file:
        for line in file:
            if line['gmap_id'] in gmap_id:
                rating = line['rating']
                total_rating += rating
                count += 1
                
                if rating <= 2:
                    line['rating_category'] = 0
                    review_low.append(line)
                elif rating >= 4:
                    line['rating_category'] = 1
                    review_high.append(line)
    
    
    return review_low, review_high



def create_tfidf_matrix(review_low, review_high):
    # Combine the processed text from review_low and review_high
    all_reviews_text = [review['processed_text'] for review in review_low] + [review['processed_text'] for review in review_high]

    # Create the TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Fit the vectorizer on the combined processed text
    vectorizer.fit(all_reviews_text)

    # Transform the processed text of review_low using the fitted vectorizer
    tfidf_matrix_low = vectorizer.transform([review['processed_text'] for review in review_low])

    # Transform the processed text of review_high using the fitted vectorizer
    tfidf_matrix_high = vectorizer.transform([review['processed_text'] for review in review_high])

    return tfidf_matrix_low, tfidf_matrix_high



# Ruta para obtener las categorías relevantes y palabras más repetidas para un local específico
@app.get("/local/")
def get_top_categories_for_local(request: Request, local_name: str):
    # Paso 1: Cargar los nombres de los locales y los gmap_id correspondientes
    local_data = []
    with open('data/strict-filtered-bars.jsonl', 'r', encoding='utf-8') as file:
        for line in file:
            entry = json.loads(line)
            local_data.append(entry)

    # Buscar el gmap_id correspondiente al nombre del local ingresado
    gmap_id = None
    for entry in local_data:
        if entry['name'].strip().lower() == local_name.strip().lower():
            gmap_id = entry['gmap_id']
            break

    # Si no se encuentra el gmap_id, mostrar un mensaje de error
    if gmap_id is None:
        return {"message": "No se encontró el local especificado."}

    # Paso 2: Cargar las reseñas y las calificaciones filtradas por el gmap_id
    reviews = []
    ratings = []

    # Cargar las reseñas y las calificaciones filtradas por el gmap_id desde el archivo JSONL
    with open('data/strict-filtered-reviews.jsonl', 'r', encoding='utf-8') as file:
        for line in file:
            entry = json.loads(line)
            if entry['gmap_id'] == gmap_id:
                reviews.append(entry['processed_text'])
                ratings.append(entry['rating'])

    # Cargar las categorías y las palabras asociadas desde el archivo lista-categorias.jsonl
    categories_data = []
    with open('data/lista-categorias.jsonl', 'r', encoding='utf-8') as file:
        for line in file:
            entry = json.loads(line)
            category_name = next(iter(entry))
            category_words = entry[category_name].split(", ")
            categories_data.append({"category": category_name, "words": category_words})

    labels = np.array([0 if rating < 3 else 1 for rating in ratings])

    X_train, X_test, y_train, y_test = train_test_split(reviews, labels, test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    # Train logistic regression model
    logreg = LogisticRegression()
    logreg.fit(X_train_vectorized, y_train)

    local_reviews = [review for review, label in zip(X_test, y_test) if label == 0]
    categories_count = {}
    top_words = {}

    # Predict the labels for local_reviews using the logistic regression model
    local_reviews_vectorized = vectorizer.transform(local_reviews)
    local_reviews_predicted = logreg.predict(local_reviews_vectorized)

    for review, predicted_label in zip(local_reviews, local_reviews_predicted):
        words = review.split()
        for category in categories_data:
            category_name = category['category']
            if category_name == "positive":
                continue

            category_words = category['words']
            count = sum(word in category_words for word in words)
            if count == 0:
                continue

            if category_name in categories_count:
                categories_count[category_name] += count
            else:
                categories_count[category_name] = count

            if category_name in top_words:
                top_words[category_name].extend([word for word in words if word in category_words])
            else:
                top_words[category_name] = [word for word in words if word in category_words]

    top_categories = sorted(categories_count.items(), key=lambda x: x[1], reverse=True)
    top_categories = [(category, count) for category, count in top_categories if category != "positive" and count > 0]

    for category, count in top_categories:
        words_count = {}
        category_words = top_words[category]
        for word in category_words:
            if word in words_count:
                words_count[word] += 1
            else:
                words_count[word] = 1

        top_words_count = sorted(words_count.items(), key=lambda x: x[1], reverse=True)[:3]
        top_words[category] = dict(top_words_count)


    review_low, review_high = get_reviews_by_gmap_ids(gmap_id)
    tfidf_matrix_low, tfidf_matrix_high = create_tfidf_matrix(review_low, review_high)



    # Combine the TF-IDF matrices
    tfidf_matrix = vstack((tfidf_matrix_low, tfidf_matrix_high))

    # Create the target variable for reviews_low and reviews_high
    target_variable_low = np.zeros(tfidf_matrix_low.shape[0])
    target_variable_high = np.ones(tfidf_matrix_high.shape[0])

    # Combine the target variables
    target_variable = np.concatenate((target_variable_low, target_variable_high))

    # Prepare your data and split into training and test sets
    X_train2, X_test2, y_train2, y_test2 = train_test_split(tfidf_matrix, target_variable, test_size=0.2, random_state=42)

    # Create and train the SVM model
    svm_model = SVR(kernel='linear')
    svm_model.fit(X_train2, y_train2)

    predicted_increases = svm_model.predict(tfidf_matrix_low)
    average_updated_rating = np.mean(predicted_increases)
    average_rating = np.mean(ratings)
    updated_ratings = average_rating + average_updated_rating
    updated_rate = np.mean(updated_ratings)

    template = env.get_template("result.html")
    html_content = template.render(
        average_rating= average_rating,
        average_updated_rating=average_updated_rating,
        updated_rate=updated_rate,
        local_name=local_name,
        top_categories=top_categories,
        top_words=top_words
    )

    return HTMLResponse(content=html_content, media_type="text/html")
