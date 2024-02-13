import os
import time
import pickle
import re
import random
import pandas as pd
import streamlit as st
from googleapiclient.discovery import build
import matplotlib.pyplot as plt
from dateutil.parser import parse
import time
import nltk
nltk.download('stopwords')

import io
from fpdf import FPDF

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from wordcloud import WordCloud



st.set_page_config(page_title='Youtube sentiment Analysis',page_icon="▶️")

hide_github_icon_js = """
<style>
#MainMenu {
    display: none;
}
button.css-ch5dnh {
    display: none;
}
</style>
<script>
document.addEventListener('DOMContentLoaded', function() {
    const toolbar = document.querySelector('[data-testid="stToolbar"]');
    if (toolbar) {
        toolbar.style.display = 'none';
    }
});
</script>
"""
st.markdown(hide_github_icon_js, unsafe_allow_html=True)


nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('vader_lexicon')

# Set up the YouTube Data API key
api_key = "AIzaSyDfojvA5tg_3YkvxasLpljV_UTB-myWHvk"
youtube = build("youtube", "v3", developerKey=api_key)

sentiment = pickle.load(open("sia.pkl",'rb'))
def extract_video_id(youtube_link):
    if "youtu.be/" in youtube_link:
        # Case 1: https://youtu.be/URyiCGZNjdI
        video_id = youtube_link.split("/")[-1].split("?")[0]
    elif "youtube.com/watch?v=" in youtube_link:
        # Case 2: https://www.youtube.com/watch?v=URyiCGZNjdI&ab_channel=melodysheep
        video_id = youtube_link.split("?v=")[1].split("&")[0]
    else:
        video_id =None

    return video_id


# Function to fetch video comments with additional information


def get_video_comments_with_info(youtube, desired_comments ,**kwargs):
    comments_data = []
    results = youtube.commentThreads().list(**kwargs).execute()

    # Fetch video details using the YouTube Data API
    video_details = youtube.videos().list(
        part="snippet", id=kwargs["videoId"]
    ).execute()
    video_item = video_details["items"][0]
    video_published_at = parse(video_item["snippet"]["publishedAt"])

    while results and len(comments_data) < desired_comments:  # Limit the number of comments retrieved
        for item in results["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            like_count = item["snippet"]["topLevelComment"]["snippet"]["likeCount"]
            published_at = item["snippet"]["topLevelComment"]["snippet"]["publishedAt"]
            author_name = item["snippet"]["topLevelComment"]["snippet"]["authorDisplayName"]
            author_id = item["snippet"]["topLevelComment"]["snippet"]["authorChannelId"]["value"]

            comment_time = parse(published_at)
            time_difference = comment_time - video_published_at
            total_replies = item["snippet"]["totalReplyCount"]


            comments_data.append({
                "Comment": comment,
                "Likes": like_count,
                "Published At": comment_time.strftime("%Y-%m-%d %H:%M:%S"),
                "Video ID": kwargs["videoId"],
                "Author Name": author_name,
                "Author ID": author_id,
                "Time Difference": str(time_difference),
                "Video Published Date and Time": video_published_at.strftime("%Y-%m-%d %H:%M:%S"),
                "Replies": total_replies

            })

            # Fetch replies for each comment
            reply_results = youtube.comments().list(
                part="snippet",
                parentId=item["snippet"]["topLevelComment"]["id"]
            ).execute()

            for reply_item in reply_results["items"]:
                reply_comment = reply_item["snippet"]["textDisplay"]
                reply_like_count = reply_item["snippet"]["likeCount"]
                reply_published_at = reply_item["snippet"]["publishedAt"]
                reply_author_name = reply_item["snippet"]["authorDisplayName"]
                reply_author_id = reply_item["snippet"]["authorChannelId"]["value"]

                reply_time = parse(reply_published_at)
                reply_time_difference = reply_time - video_published_at

                comments_data.append({
                    "Comment": reply_comment,
                    "Likes": reply_like_count,
                    "Published At": reply_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "Video ID": kwargs["videoId"],
                    "Author Name": reply_author_name,
                    "Author ID": reply_author_id,
                    "Time Difference": str(reply_time_difference),
                    "Video Published Date and Time": video_published_at.strftime("%Y-%m-%d %H:%M:%S"),
                    "Replies": 0  # Set 0 for replies to replies

                })

            if len(comments_data) >= desired_comments:
                break
        # Check for the presence of more comments
        if "nextPageToken" in results and len(comments_data) < desired_comments:
            kwargs["pageToken"] = results["nextPageToken"]
            results = youtube.commentThreads().list(**kwargs).execute()
        else:
            break

    return comments_data


# Function to fetch video details using the YouTube Data API
def get_video_details(youtube, video_id):
    video_details = youtube.videos().list(
        part="snippet,statistics",
        id=video_id
    ).execute()
    
    if "items" in video_details:
        video_item = video_details["items"][0]
        details = {
            "Title": video_item["snippet"]["title"],
            "Description": video_item["snippet"]["description"],
            "Views": video_item["statistics"]["viewCount"],
            "Likes": video_item["statistics"]["likeCount"],
            # "Dislikes": video_item["statistics"]["dislikeCount"]
        }
        return details
    else:
        return None


def save_as_pdf():
    # Create a PDF instance
    pdf = FPDF()
    pdf.add_page()

    # Add title
    pdf.set_font("Arial", size = 12)
    pdf.cell(200, 10, txt = "YouTube Sentiment Analysis", ln = True, align = 'C')

    # Add content from Streamlit
    for element in st.container_list:
        if element is not None:
            # Write Streamlit element to PDF
            if isinstance(element, str):
                pdf.cell(200, 10, txt = element, ln = True)
            elif hasattr(element, 'write'):
                # Capture matplotlib plots and write to PDF
                plt = element.pyplot()
                if plt is not None:
                    plt.savefig("temp_plot.png", format="png")
                    pdf.image("temp_plot.png", x = None, y = None, w = 0, h = 0)
            elif hasattr(element, 'data'):
                # Capture WordCloud and write to PDF
                wordcloud = element.data[0].get("image/png")
                if wordcloud is not None:
                    with open("temp_wordcloud.png", "wb") as f:
                        f.write(wordcloud)
                    pdf.image("temp_wordcloud.png", x = None, y = None, w = 0, h = 0)

    # Save PDF
    pdf_output = "youtube_sentiment_analysis.pdf"
    pdf.output(pdf_output)

    # Inform user
    st.success(f"PDF saved as {pdf_output}")





def main():
    colIm,coltit = st.columns([1,3])
    colIm.image("3721679-youtube_108064.png",width=160)
    coltit.title("YouTube Sentiment Analysis")
    st.sidebar.title("About this webApp")
    st.sidebar.write("The YouTube Sentiment Analysis web app allows users to analyze the sentiment distribution of comments on a given YouTube video. By entering the video link and selecting the percentage of comments to analyze. It performs sentiment analysis to categorize comments as positive, negative, or neutral. The app perform sentiment analysis,engagement analysis and offers time analysis for viewer engagement.  it provides valuable insights for content creators and marketers.")
    st.toast("Loading...🤓")
    

    # Text input for video link
    youtube_link = st.text_input("Enter YouTube Video Link:", "")
    st.markdown(
        "Made by Pranav And Radha | [LinkedIn](https://www.linkedin.com/in/pranav-chouhan-a14399207/)")


    # Slider to select the percentage of comments to retrieve
    selected_percentage = st.slider('percents of comments to be analyze:',20, 70,step=10)
    # Display video details section
    if youtube_link:
        video_id = extract_video_id(youtube_link)
        if video_id:
            video_details = get_video_details(youtube, video_id)
            if video_details:
                st.write("### Video Details")
                st.write(f"**Title:** {video_details['Title']}")
                st.write(f"**Description:** {video_details['Description']}")
                st.write(f"**Views:** {video_details['Views']}")
                st.write(f"**Likes:** {video_details['Likes']}")
                # st.write(f"**Dislikes:** {video_details['Dislikes']}")
            else:
                st.write("Unable to fetch video details.")
        else:
            st.write("Invalid YouTube Video Link. Please enter a valid link.")


    if youtube_link:
        # Extract the video ID from the link
        video_id = extract_video_id(youtube_link)

        if video_id:
            # Fetch video details using the YouTube Data API
            video_details = youtube.videos().list(
                part="statistics", id=video_id
            ).execute()

            # Get the total number of comments
            if "items" in video_details:
                video_item = video_details["items"][0]
                comment_count = int(video_item["statistics"]["commentCount"])
            else:
                comment_count = 0

            # Calculate the desired number of comments based on selected percentage
            desired_comments = int(comment_count * selected_percentage / 100)

            # Limit the number of comments to retrieve based on the desired_comments count
            if desired_comments > 0:
                try:
                        with st.spinner(f"hold on! bro ,Loading {comment_count} comments.😄.."):
                            start_time = time.time()
                            st.write("fetching time directly depends on number of comments and your internet connectivity")


                            comments_data = get_video_comments_with_info(
                                youtube,
                                desired_comments,
                                part="snippet",
                                videoId=video_id,
                                textFormat="plainText"
                            )
                            end_time = time.time()

                        # Calculate the total time taken to retrieve all comments
                        total_time_taken = end_time - start_time


                        comments_df = pd.DataFrame(comments_data)

                        comments_df['Comment'] = comments_df['Comment'].apply(remove_punctuation)
                        comments_df['Comment'] = comments_df['Comment'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 3]))
                        comments_df['Comment'] = comments_df['Comment'].apply(lambda x: x.lower())


                        comments_df['Comment'] = comments_df['Comment'].apply(lemmatize_and_join)

                        comments_df['Sentiment Scores'] = comments_df['Comment'].apply(
                            lambda x: sentiment.polarity_scores(x)['compound'])
                        comments_df['Sentiment'] = comments_df['Sentiment Scores'].apply(
                            lambda s: 'Positive' if s > 0 else ('Neutral' if s == 0 else 'Negative'))
                        sentiment_counts = comments_df['Sentiment'].value_counts()

                        col1, col2 = st.columns(2)

                        total_comments = sentiment_counts.sum()
                        percentage_positive = (sentiment_counts['Positive'] / total_comments) * 100
                        percentage_neutral = (sentiment_counts['Neutral'] / total_comments) * 100
                        percentage_negative = 100 - percentage_positive - percentage_neutral
                        col1.write('Sentiment Distribution')
                        plt.style.use('dark_background')
                        # Plot the pie chart
                        labels = ['Positive', 'Neutral', 'Negative']
                        sizes = [percentage_positive, percentage_neutral, percentage_negative]
                        colors = ['#00adb5', '#ff6b6b', '#ffd166']

                        # Explode the Positive slice to make it stand out
                        explode = (0.1, 0, 0)

                        # Plot the pie chart with a black background and other customizations
                        plt.style.use('dark_background')

                        plt.figure(figsize=(8, 5))
                        plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True,
                                startangle=140,textprops={'color': 'black'})
                        plt.axis('equal')

                        # Set text color to white and pie chart edgecolor to black
                        plt.rcParams['text.color'] = 'white'
                        for wedge in plt.gca().patches:
                            wedge.set_edgecolor('black')

                        # Set the title and legend
                        plt.title('Sentiment Distribution of Comments', color='white', fontsize=16)
                        plt.legend(labels, loc='upper right', bbox_to_anchor=(1, 0.9), fontsize=12)

                        # Add a shadow to the pie chart
                        # plt.gca().add_artist(plt.Circle((0, 0), 0.5, color='black', fill=False))

                        # Remove the black wedge between the slices
                        plt.gca().patches[1].set_visible(False)

                        col1.pyplot(plt)

                        # Display the most liked and most replied comments
                        col2.write(f"Total Number of Comments on the Video: {comment_count}")
                        col2.write(f"Number of Comments Fetched: {len(comments_data)}")
                        col2.write(f"Total Time Taken: {total_time_taken:.2f} seconds")

                        # Find and display the most liked comment
                        most_liked_comment = comments_df.iloc[comments_df['Likes'].idxmax()]['Comment']
                        most_likes = comments_df['Likes'].max()
                        col2.write(f"Most Liked Comment ({most_likes} likes):")
                        col2.write(f'"->{most_liked_comment}"')

                        # Find and display the most replied comment
                        most_replied_comment = comments_df.iloc[comments_df['Replies'].idxmax()]['Comment']
                        most_replies = comments_df['Replies'].max()
                        col2.write(f"Most Replied Comment ({most_replies} replies):")
                        col2.write(f'"->{most_replied_comment}"')


                        total_video_likes = comments_df['Likes'].sum()

                        # Calculate the engagement rate for each comment and add it to a new column 'Engagement Rate'
                        comments_df['Engagement Rate'] = (comments_df['Likes'] / total_video_likes) * 100

                        # Calculate the average engagement rate for each sentiment
                        average_engagement_by_sentiment = comments_df.groupby('Sentiment')['Engagement Rate'].mean()

                        # Plot the bar chart to visualize the relationship between sentiment and average engagement rate
                        plt.figure(figsize=(8, 5))
                        average_engagement_by_sentiment.plot(kind='bar', color=['#66bb6a', '#ffca28', '#ef5350'])
                        plt.xticks(rotation=0)
                        plt.xlabel('Sentiment')
                        plt.ylabel('Average Engagement Rate (%)')
                        plt.title('Relationship between Sentiment and Engagement')
                        plt.grid(axis='y')
                        plt.tight_layout()

                        # Display the bar chart in Streamlit
                        st.pyplot(plt)

                        # Sentiment Over Time
                        # Convert 'Published At' to datetime format
                        comments_df['Published At'] = pd.to_datetime(comments_df['Published At'])

                        # Group by 'Published At' and calculate the average sentiment score for each time period
                        sentiment_over_time = comments_df.groupby(pd.Grouper(key='Published At', freq='1H'))[
                            'Sentiment Scores'].mean().reset_index()

                        # Plot sentiment trend over time
                        plt.figure(figsize=(10, 6))
                        plt.plot(sentiment_over_time['Published At'], sentiment_over_time['Sentiment Scores'], marker='o',
                                 color='dodgerblue', linestyle='-')
                        plt.xlabel('Time')
                        plt.ylabel('Average Sentiment Score')
                        plt.title('Sentiment Trend Over Time')
                        plt.grid(True)
                        plt.tight_layout()

                        # Display the plot
                        st.pyplot(plt)

                        comments_df['Time Difference (minutes)'] = (comments_df[
                                                                        'Published At'] - comments_df['Published At'].min()).dt.total_seconds() / 60

                        # Group the DataFrame by time difference in minutes and count the number of comments at each time point
                        time_grouped = comments_df.groupby('Time Difference (minutes)').size().reset_index(
                            name='Number of Comments')

                        # Calculate the cumulative sum of comments
                        time_grouped['Cumulative Comments'] = time_grouped['Number of Comments'].cumsum()

                        # Plot the time analysis with the cumulative number of comments
                        plt.figure(figsize=(10, 6))
                        plt.plot(time_grouped['Time Difference (minutes)'], time_grouped['Cumulative Comments'], marker='o',
                                 markersize=6, color='dodgerblue', linestyle='-', linewidth=1, alpha=0.7)
                        plt.xlabel('Time Difference from Video Published Time (minutes)')
                        plt.ylabel('Cumulative Number of Comments')
                        plt.title('Time Analysis of Cumulative Number of Comments')
                        plt.grid(True)
                        plt.xticks(range(0, 61, 5))  # Set x-axis ticks to display every 5 minutes from 0 to 60
                        plt.xlim(0, 60)  # Set x-axis limit to 60 minutes (1 hour)
                        plt.tight_layout()

                        # Show the plot
                        st.pyplot(plt)

                        # Function to create word clouds for each sentiment category
                        def create_word_cloud(sentiment_category):
                            comments_text = ' '.join(comments_df[comments_df['Sentiment'] == sentiment_category]['Comment'])
                            if not comments_text:
                                st.write(f"No {sentiment_category} comments found.")
                                return

                            wordcloud = WordCloud(width=800, height=400, background_color='black').generate(comments_text)
                            plt.figure(figsize=(10, 5))
                            plt.imshow(wordcloud, interpolation='bilinear')
                            plt.axis('off')
                            plt.title(f'Word Cloud for {sentiment_category} Comments')
                            st.pyplot(plt)

                        # Create word clouds for positive, negative, and neutral comments
                        create_word_cloud('Positive')
                        create_word_cloud('Negative')
                        create_word_cloud('Neutral')

                        if st.button("Save as PDF"):
                            save_as_pdf()
                                            

                            
                except Exception as e:
                    st.error("Failed to retrieve comments.  YouTube API quota exceeded , try tomoroow.")


            else:
                st.write("No comments available for the selected percentage.")
        else:
             st.write("Invalid YouTube Video Link. Please enter a valid link.")

def remove_punctuation(text):
    cleaned_text = re.sub(r"[^a-zA-Z#]", " ",text)
    return cleaned_text

wnl = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def lemmatize_and_join(text):
    words = text.split()
    wnl = WordNetLemmatizer()
    lemmatized_words = [wnl.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(lemmatized_words)


    # Disable the warning about global usage of pyplot
st.set_option('deprecation.showPyplotGlobalUse', False)

if __name__ == "__main__":
    main()

