import streamlit as st
import pickle

from textblob import TextBlob

from wordcloud import WordCloud
import matplotlib.pyplot as plt

model = pickle.load(
    open("model.pkl","rb")
)

vectorizer = pickle.load(
    open("vectorizer.pkl","rb")
)

st.set_page_config(
    page_title="Review Intelligence Dashboard",
    page_icon="🔍",
    layout="wide"
)

st.title(
    "🔍 Review Intelligence Dashboard"
)

st.markdown(
    "AI-Powered Fake Review Detection and Analysis"
)

review = st.text_area(
    "Enter Review"
)

examples = [
    "Amazing product must buy now!",
    "Good quality and fast delivery",
    "Worst product I have ever purchased"
]

selected = st.selectbox(
    "Try Sample Review",
    [""] + examples
)

if selected:
    review = selected

if st.button("Analyze"):

    transformed = vectorizer.transform(
        [review]
    )

    prediction = model.predict(
        transformed
    )[0]

    probability = max(model.predict_proba(transformed)[0])


    if prediction == "real":

        st.success(
            f"Genuine Review ({probability*100:.2f}%)"
        )

    else:

        st.error(
            f"Fake Review ({probability*100:.2f}%)"
        )

    confidence = probability * 100

    st.metric(
        "Confidence Score",
        f"{confidence:.2f}%"
    )

    st.progress(int(confidence))   

    word_count = len(review.split())
    char_count = len(review)

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Words", word_count)

    with col2:
        st.metric("Characters", char_count)     

    blob = TextBlob(review)
    sentiment = blob.sentiment.polarity

    if sentiment > 0:
        st.success("😊 Positive Review")
    elif sentiment < 0:
        st.error("😠 Negative Review")
    else:
        st.info("😐 Neutral Review") 

    suspicious_words = [
        "best",
        "amazing",
        "must buy",
        "perfect",
        "unbelievable",
        "guaranteed"
    ]

    found = []

    for word in suspicious_words:
        if word in review.lower():
            found.append(word)\

    if found:
        st.warning(
            "Suspicious phrases: "
            + ", ".join(found)
        )  

    wc = WordCloud(
        width=800,
        height=400
    ).generate(review)

    fig, ax = plt.subplots()

    ax.imshow(wc)

    ax.axis("off")

    st.pyplot(fig)                             

