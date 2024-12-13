import streamlit as st
import arxiv
import pandas as pd
import plotly.express as px
import nltk
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances, silhouette_score
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Загрузка необходимых компонентов для VADER и NLTK
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Функция для получения публикаций с arXiv
def get_arxiv_data(query, max_results=100):
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    papers = []
    try:
        for result in search.results():
            papers.append({
                'title': result.title,
                'summary': result.summary,
                'url': result.entry_id
            })
    except Exception as e:
        st.error(f"Ошибка при получении данных с arXiv: {e}")
        return []

    return papers

# Функция для предобработки текста (токенизация, лемматизация, удаление стоп-слов)
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())  # Токенизация и приведение к нижнему регистру
    filtered_words = [lemmatizer.lemmatize(word) for word in words if word.isalpha() and word not in stop_words]  # Удаление стоп-слов и лемматизация
    return " ".join(filtered_words)

# Функция для получения BERT-эмбеддингов
def get_bert_embeddings(texts):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).numpy()  # Среднее значение по всем токенам для получения эмбеддинга
    return embeddings

# Функция для выполнения тематического моделирования с использованием LDA, NMF или BERT
def topic_modeling(docs, method='LDA', n_topics=5):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(docs)

    if method == 'LDA':
        model = LDA(n_components=n_topics, random_state=42)
        model.fit(tfidf_matrix)
        topics = model.components_
    elif method == 'NMF':
        model = NMF(n_components=n_topics, random_state=42)
        model.fit(tfidf_matrix)
        topics = model.components_
    elif method == 'BERT':
        embeddings = get_bert_embeddings(docs)
        pairwise_dist = pairwise_distances(embeddings, metric='cosine')
        return pairwise_dist

    # Получаем топ-10 слов для каждой темы
    feature_names = np.array(vectorizer.get_feature_names_out())
    topics_words = []
    for topic in topics:
        top_words_idx = topic.argsort()[:-11:-1]
        top_words = feature_names[top_words_idx]
        topics_words.append(" ".join(top_words))

    return topics_words

# Интерфейс с Streamlit
st.title('Тематическое моделирование научных статей по Data Science')

# Ввод темы для поиска статей
query = st.text_input('Введите тему для поиска статей', 'Data Science')

# Получаем данные с arXiv
max_results = st.slider('Количество статей для анализа', 5, 50, 10)
papers = get_arxiv_data(query, max_results)

if papers:  # Если данные успешно получены
    # Преобразуем в DataFrame для удобной работы
    df = pd.DataFrame(papers)

    # Отображаем таблицу с результатами
    st.write(df[['title', 'summary', 'url']])

    # Выполняем тематическое моделирование
    docs = df['summary'].apply(preprocess_text).tolist()
    st.subheader('Выберите метод тематического моделирования')
    method = st.radio('Метод', ('LDA', 'NMF', 'BERT'))

    n_topics = st.slider('Количество тем', 2, 10, 5)
    topics_words = topic_modeling(docs, method=method, n_topics=n_topics)

    if method != 'BERT':
        st.subheader('Топ-10 слов для каждой темы')
        for i, words in enumerate(topics_words):
            st.write(f"Тема {i+1}: {words}")
    else:
        st.subheader('Косинусное расстояние между статьями (по BERT)')
        st.write(topics_words)  # Выводим результаты косинусных расстояний

    # Оценка качества темы с использованием Silhouette Score для LDA и NMF
    if method in ['LDA', 'NMF']:
        model = LDA(n_components=n_topics, random_state=42) if method == 'LDA' else NMF(n_components=n_topics, random_state=42)
        model.fit(TfidfVectorizer(stop_words='english').fit_transform(docs))
        topic_matrix = model.transform(TfidfVectorizer(stop_words='english').fit_transform(docs))
        score = silhouette_score(topic_matrix, np.argmax(topic_matrix, axis=1))
        st.write(f"Качество модели (Silhouette Score): {score:.2f}")

    # Визуализация результатов тематического моделирования
    st.subheader('Карта тем (с помощью NMF или LDA)')
    if method != 'BERT':
        model = LDA(n_components=n_topics, random_state=42) if method == 'LDA' else NMF(n_components=n_topics, random_state=42)
        model.fit(TfidfVectorizer(stop_words='english').fit_transform(docs))
        topic_matrix = model.transform(TfidfVectorizer(stop_words='english').fit_transform(docs))

        # Визуализация с использованием PCA для снижения размерности
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(topic_matrix)

        df_topics = pd.DataFrame(pca_result, columns=['x', 'y'])
        df_topics['topic'] = np.argmax(topic_matrix, axis=1)

        fig = px.scatter(df_topics, x='x', y='y', color='topic', title="Распределение статей по темам")
        st.plotly_chart(fig)
else:
    st.write("Нет данных для отображения.")
