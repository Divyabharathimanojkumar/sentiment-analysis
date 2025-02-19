import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def analyze_text_sentiment(text):
    # Use TextBlob to compute polarity and subjectivity
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    textblob_sentiment = "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral"
    
    # Use VADER for sentiment scores
    analyzer = SentimentIntensityAnalyzer()
    vader_scores = analyzer.polarity_scores(text)
    
    return {
        "text": text,
        "textblob": {
            "polarity": polarity,
            "subjectivity": subjectivity,
            "sentiment": textblob_sentiment
        },
        "vader": vader_scores
    }

if __name__ == '__main__':
    # Replace with your Kaggle CSV dataset file name
    csv_file = "training.1600000.processed.noemoticon.csv"
      # e.g., "sentiment140.csv"
    csv_file = "Tweets.csv"
    
    # Load the dataset into a pandas DataFrame
    # Ensure the dataset has a column (here assumed to be named "text") that contains the text for analysis
    df = pd.read_csv(csv_file)
    
    # Apply sentiment analysis to each row's text
    # If your column is named differently (e.g., "tweet" or "review"), update 'text' accordingly.
    df['analysis'] = df['text'].apply(analyze_text_sentiment)
    
    # Normalize the nested dictionary structure into separate columns
    analysis_df = pd.json_normalize(df['analysis'])
    
    # Merge the analysis results back into the original DataFrame
    df = pd.concat([df, analysis_df], axis=1)
    
    # Display the first few rows of the resulting DataFrame
    print(df.head())
    
    # Optionally, save the results to a new CSV file
    output_file = "sentiment_analysis_results.csv"
    df.to_csv(output_file, index=False)
    print(f"Sentiment analysis results saved to {output_file}")
