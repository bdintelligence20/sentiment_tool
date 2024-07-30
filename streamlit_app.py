import streamlit as st
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px
import base64
import io
from openai import OpenAI
import textwrap
import time
from dotenv import load_dotenv
import os

load_dotenv()  # take environment variables from .env.

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OpenAI API Key not found in environment variables.")

client = OpenAI(api_key=openai_api_key)

# Initialize VADER sentiment analyzer
vader_analyzer = SentimentIntensityAnalyzer()

@st.cache_data
def load_csv(file_path):
    return pd.read_csv(file_path)

@st.cache_data
def perform_sentiment_analysis(df, text_columns):
    for text_column in text_columns:
        if text_column not in df.columns:
            raise KeyError(f"Column '{text_column}' not found in the CSV file. Available columns: {df.columns.tolist()}")
        df[f'VADER_Sentiment_{text_column}'] = df[text_column].apply(lambda x: vader_analyzer.polarity_scores(str(x))['compound'])
        df[f'VADER_Sentiment_Label_{text_column}'] = df[f'VADER_Sentiment_{text_column}'].apply(lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral'))
    return df

@st.cache_data
def combine_text(df, text_columns):
    return " ".join(df[text_columns].fillna('').values.flatten())

@st.cache_data
def generate_word_cloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close(fig)
    
    return base64.b64encode(img.getvalue()).decode()

@st.cache_data
def generate_sentiment_bar_graph(df, text_columns):
    sentiment_data = pd.DataFrame()
    
    for text_column in text_columns:
        sentiment_label_col = f'VADER_Sentiment_Label_{text_column}'
        if sentiment_label_col in df.columns:
            sentiment_counts = df[sentiment_label_col].value_counts().reset_index()
            sentiment_counts.columns = ['Sentiment', 'Count']
            sentiment_data = pd.concat([sentiment_data, sentiment_counts], axis=0)

    sentiment_data = sentiment_data.groupby('Sentiment').sum().reset_index()

    fig = px.bar(sentiment_data, x='Sentiment', y='Count', color='Sentiment',
                 title='Overall Sentiment Analysis Results', height=800)

    fig.update_layout(
        title_font_size=24, xaxis_title_font_size=18, yaxis_title_font_size=18,
        legend_title_font_size=18, legend_font_size=14,
        xaxis={'categoryorder': 'total descending', 'title': {'text': ''}},
        yaxis={'title': {'text': 'Count'}, 'tickmode': 'linear', 'dtick': 2},
        margin=dict(l=20, r=20, t=40, b=150),
        height=800,
        width=1200,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig

@st.cache_data
def get_gpt4_insights(combined_text):
    max_retries = 3
    retry_delay = 5

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
  model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a business analyst. Your job is to parse through the provided text and understand the content within the selected columns. You then must provide 12 focus areas of improvement. These areas are based on what is within the content and needs to be a reflection of the feedback."
                    },
                    {
                        "role": "user",
                        "content": combined_text
                    }
                ],
                temperature=1,
                max_tokens=4095,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                st.warning(f"API connection error. Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
            else:
                st.error("Failed to connect to the OpenAI API after multiple attempts. Please try again later.")
                return "Unable to generate insights due to API connection issues."

def main():
    st.set_page_config(layout="wide")
    st.title("Sentiment Analysis and Word Cloud Generator with GPT-4 Insights")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        data = load_csv(uploaded_file)

        text_columns = st.multiselect(
            "Select text columns for analysis",
            data.columns.tolist()
        )

        if st.button("Perform Sentiment Analysis"):
            try:
                data_with_sentiment = perform_sentiment_analysis(data, text_columns)
                
                sentiment_counts = {text_column: data_with_sentiment[f'VADER_Sentiment_Label_{text_column}'].value_counts().to_dict() for text_column in text_columns}

                st.write("### Sentiment Counts for each column:")
                for column, counts in sentiment_counts.items():
                    st.write(f"**{column}:**")
                    st.write(pd.DataFrame.from_dict(counts, orient='index', columns=['Count']))

                combined_text = combine_text(data_with_sentiment, text_columns)
                
                st.subheader("Word Cloud")
                wordcloud_base64 = generate_word_cloud(combined_text)
                st.image(f"data:image/png;base64,{wordcloud_base64}", use_column_width=True)

                st.subheader("Sentiment Analysis Bar Graph")
                sentiment_bar_graph = generate_sentiment_bar_graph(data_with_sentiment, text_columns)
                st.plotly_chart(sentiment_bar_graph, use_container_width=True)

                st.subheader("Key Takeways and Insights From Data")
                insights = get_gpt4_insights(combined_text)
                st.write(insights)
                
            except KeyError as e:
                st.error(e)

if __name__ == "__main__":
    main()
