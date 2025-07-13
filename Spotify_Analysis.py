import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import requests
from io import StringIO

# Set page config
st.set_page_config(
    page_title="Spotify Music Analysis",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data function
@st.cache_data
def load_data():
    # Replace with your GitHub raw CSV URL
    github_url = "https://raw.githubusercontent.com/yourusername/yourrepo/main/yourdata.csv"
    
    try:
        response = requests.get(github_url)
        response.raise_for_status()
        data = pd.read_csv(StringIO(response.text))
        
        # Data cleaning and preprocessing
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'])
            data['Year'] = data['Date'].dt.year
            data['Month'] = data['Date'].dt.month_name()
            data['DayOfWeek'] = data['Date'].dt.day_name()
        
        if 'Streams' in data.columns:
            # Convert streams to millions for better readability
            data['Streams_M'] = data['Streams'] / 1000000
            
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Machine learning functions
def perform_clustering(data, n_clusters=4):
    """Perform K-means clustering on the data"""
    try:
        # Select numerical features
        features = data.select_dtypes(include=['float64', 'int64'])
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(features_scaled)
        
        return clusters
    except Exception as e:
        st.error(f"Clustering error: {e}")
        return None

def predict_trends(data):
    """Simple trend prediction using moving averages"""
    try:
        if 'Date' in data.columns and 'Streams' in data.columns:
            ts_data = data.set_index('Date')['Streams'].sort_index()
            moving_avg = ts_data.rolling(window=30).mean()
            return moving_avg
        return None
    except Exception as e:
        st.error(f"Trend prediction error: {e}")
        return None

# Main app
def main():
    st.title("🎵 Spotify Music Analysis Dashboard")
    st.markdown("""
    This interactive dashboard provides insights into Spotify streaming data, 
    including trends, artist performance, and predictive analytics.
    """)
    
    # Load data
    data = load_data()
    
    if data is not None:
        # Overview section
        st.header("📊 Overview")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_streams = data['Streams'].sum() / 1000000000  # Convert to billions
            st.metric("Total Streams", f"{total_streams:.2f} Billion")
        
        with col2:
            avg_streams = data['Streams'].mean() / 1000000  # Convert to millions
            st.metric("Average Streams per Track", f"{avg_streams:.2f} Million")
        
        with col3:
            unique_tracks = data['Track Name'].nunique()
            st.metric("Unique Tracks", f"{unique_tracks:,}")
        
        # Power BI Embed Section
        st.header("📈 Interactive Power BI Dashboard")
        st.markdown("""
        Below is the embedded Power BI dashboard for interactive exploration of the Spotify data.
        """)
        
        # Replace with your Power BI embed URL
        powerbi_url = "https://app.powerbi.com/reportEmbed?reportId=your-report-id&groupId=your-group-id"
        st.components.v1.iframe(powerbi_url, width=1000, height=600, scrolling=True)
        
        # Advanced Analytics Section
        st.header("🔍 Advanced Analytics")
        
        # Tab layout for different analyses
        tab1, tab2, tab3 = st.tabs(["Trend Analysis", "Artist Insights", "Machine Learning"])
        
        with tab1:
            st.subheader("Streaming Trends Over Time")
            
            if 'Date' in data.columns and 'Streams' in data.columns:
                # Monthly trends
                monthly_data = data.groupby(['Year', 'Month'])['Streams'].sum().reset_index()
                fig = px.line(monthly_data, x='Month', y='Streams', color='Year',
                              title="Monthly Streaming Trends by Year")
                st.plotly_chart(fig, use_container_width=True)
                
                # Trend prediction
                st.subheader("30-Day Moving Average Trend Prediction")
                moving_avg = predict_trends(data)
                if moving_avg is not None:
                    fig = px.line(moving_avg.reset_index(), x='Date', y='Streams',
                                 title="Streaming Trend with 30-Day Moving Average")
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Artist Performance Analysis")
            
            if 'Artist' in data.columns:
                top_artists = data.groupby('Artist')['Streams'].sum().nlargest(10).reset_index()
                fig = px.bar(top_artists, x='Artist', y='Streams', 
                             title="Top 10 Artists by Total Streams")
                st.plotly_chart(fig, use_container_width=True)
                
                # Artist clustering
                st.subheader("Artist Clustering by Performance")
                artist_stats = data.groupby('Artist').agg({
                    'Streams': ['sum', 'mean', 'count'],
                    'Track Name': 'nunique'
                }).reset_index()
                artist_stats.columns = ['Artist', 'Total_Streams', 'Avg_Streams', 
                                       'Track_Count', 'Unique_Tracks']
                
                clusters = perform_clustering(artist_stats.select_dtypes(include=['float64', 'int64']))
                if clusters is not None:
                    artist_stats['Cluster'] = clusters
                    fig = px.scatter(artist_stats, x='Total_Streams', y='Avg_Streams',
                                     color='Cluster', hover_name='Artist',
                                     title="Artist Clustering by Streaming Performance")
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Machine Learning Insights")
            
            # Track clustering
            st.markdown("### Track Clustering by Features")
            if all(col in data.columns for col in ['Streams', 'Mode', 'Key']):
                track_features = data[['Streams', 'Mode', 'Key']]
                track_features = pd.get_dummies(track_features, columns=['Mode', 'Key'])
                
                clusters = perform_clustering(track_features)
                if clusters is not None:
                    data['Cluster'] = clusters
                    
                    cluster_summary = data.groupby('Cluster').agg({
                        'Streams': ['mean', 'count'],
                        'Mode': lambda x: x.mode()[0],
                        'Key': lambda x: x.mode()[0]
                    }).reset_index()
                    
                    st.dataframe(cluster_summary)
                    
                    fig = px.box(data, x='Cluster', y='Streams', 
                                 title="Stream Distribution by Cluster")
                    st.plotly_chart(fig, use_container_width=True)
            
            # Recommendation system placeholder
            st.markdown("### Similar Track Recommendation")
            track_name = st.selectbox("Select a track to find similar ones:", 
                                     data['Track Name'].unique())
            
            # This is a placeholder - you would implement actual recommendation logic
            if track_name:
                st.info("Recommendation system would show tracks with similar features here.")
    
    else:
        st.error("Failed to load data. Please check the data source.")

if __name__ == "__main__":
    main()
