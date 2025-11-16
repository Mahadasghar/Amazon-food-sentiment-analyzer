import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re
import nltk

# Download required NLTK data
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# Page config
st.set_page_config(
    page_title="Amazon Food Review Sentiment Analyzer",
    page_icon="🍔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []

# Load models
# Load models
@st.cache_resource
def load_models():
    """Load VADER and fine-tuned RoBERTa models"""
    # VADER
    sia = SentimentIntensityAnalyzer()
    
    # Fine-tuned RoBERTa (YOUR MODEL - 83% accuracy!)
    MODEL_PATH = "final_model_for_streamlit"  # ← REMOVED the "./"
    
    try:
        # Use local_files_only=True to force loading from local directory
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            local_files_only=True
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_PATH,
            local_files_only=True
        )
        
        # Move to appropriate device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()  # Set to evaluation mode
        
        st.success(f" Models loaded successfully! Device: {device}")
        
        return sia, tokenizer, model, device
    except Exception as e:
        st.error(f"Error loading fine-tuned model: {e}")
        st.info("Make sure 'final_model_for_streamlit' folder is in the same directory as app.py")
        st.info(f"Current working directory: {os.getcwd()}")
        
        # Show what files are in current directory
        import os
        if os.path.exists("final_model_for_streamlit"):
            st.success(" Model folder found!")
            files = os.listdir("final_model_for_streamlit")
            st.write("Files in model folder:", files)
        else:
            st.error(" Model folder NOT found!")
            st.write("Available folders:", [f for f in os.listdir(".") if os.path.isdir(f)])
        
        return None, None, None, None


# Clean text function
def clean_text(text):
    text = str(text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = ' '.join(text.split())
    return text

# Prediction functions
def get_vader_prediction(text, sia):
    """Get VADER sentiment scores"""
    scores = sia.polarity_scores(text)
    sentiment = max(scores, key=lambda x: scores[x] if x != 'compound' else -1)
    if sentiment == 'neg':
        sentiment = 'negative'
    elif sentiment == 'neu':
        sentiment = 'neutral'
    elif sentiment == 'pos':
        sentiment = 'positive'
    return {
        'sentiment': sentiment,
        'negative': scores['neg'],
        'neutral': scores['neu'],
        'positive': scores['pos'],
        'compound': scores['compound']
    }

def get_roberta_prediction(text, tokenizer, model, device):
    """Get fine-tuned RoBERTa sentiment scores (83% accuracy)"""
    # Tokenize with max_length=256 (as used in training)
    encoded = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        max_length=256,
        padding='max_length'
    ).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(**encoded)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # Move to CPU and convert to numpy
    scores = probs[0].cpu().numpy()
    
    sentiments = ['negative', 'neutral', 'positive']
    sentiment = sentiments[np.argmax(scores)]
    
    return {
        'sentiment': sentiment,
        'negative': float(scores[0]),
        'neutral': float(scores[1]),
        'positive': float(scores[2])
    }

def get_ensemble_prediction(vader_scores, roberta_scores):
    """Get ensemble prediction using weighted average (based on actual accuracies)"""
    # Updated weights based on your actual performance
    # VADER: 17%, RoBERTa: 83%
    vader_weight = 0.17
    roberta_weight = 0.83
    
    neg_score = (vader_scores['negative'] * vader_weight) + (roberta_scores['negative'] * roberta_weight)
    neu_score = (vader_scores['neutral'] * vader_weight) + (roberta_scores['neutral'] * roberta_weight)
    pos_score = (vader_scores['positive'] * vader_weight) + (roberta_scores['positive'] * roberta_weight)
    
    scores = {'negative': neg_score, 'neutral': neu_score, 'positive': pos_score}
    sentiment = max(scores, key=scores.get)
    
    return {
        'sentiment': sentiment,
        'negative': neg_score,
        'neutral': neu_score,
        'positive': pos_score
    }

# Main app
def main():
    # Header
    st.title("🍔 Amazon Food Review Sentiment Analyzer")
    st.markdown("### Fine-tuned RoBERTa Model (83% Test Accuracy)")
    
    # Load models
    with st.spinner("Loading AI models..."):
        result = load_models()
        if result[0] is None:
            st.stop()
        sia, tokenizer, model, device = result
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        mode = st.radio(
            "Select Mode:",
            ["Single Review", "Batch Analysis", "View History"],
            help="Choose between analyzing a single review or multiple reviews"
        )
        
        st.markdown("---")
        st.markdown("### 📊 Model Performance")
        st.info("""
        **Trained on 50,000 Amazon Food Reviews**
        """)
        st.metric("VADER (Baseline)", "17%", help="Rule-based model")
        st.metric("RoBERTa (Pre-trained)", "67%", help="Before fine-tuning")
        st.metric("RoBERTa (Fine-tuned)", "83%", delta="+16%", help="After fine-tuning")
        st.metric("Ensemble", "~84%", delta="+1%", help="Weighted combination")
        
        st.markdown("---")
        st.markdown("### 🎯 About Models")
        st.info("""
        - **VADER**: Fast rule-based model
        - **RoBERTa**: Fine-tuned transformer (your custom model!)
        - **Ensemble**: Weighted combination (83% RoBERTa + 17% VADER)
        """)
        
        st.markdown("---")
        st.markdown("### 📈 Training Details")
        st.caption(f"""
        - Base Model: cardiffnlp/twitter-roberta-base-sentiment
        - Fine-tuning Epochs: 5
        - Training Samples: 40,000
        - Test Samples: 10,000
        - Max Length: 256 tokens
        - Device: {device}
        """)
    
    # Main content based on mode
    if mode == "Single Review":
        single_review_mode(sia, tokenizer, model, device)
    elif mode == "Batch Analysis":
        batch_analysis_mode(sia, tokenizer, model, device)
    else:
        view_history_mode()

def single_review_mode(sia, tokenizer, model, device):
    """Single review analysis interface"""
    st.header("📝 Single Review Analysis")
    
    # Example reviews
    st.markdown("**Try these examples:**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🟢 Positive Example", use_container_width=True):
            st.session_state.example_text = "This coffee is absolutely amazing! Rich flavor, perfect aroma, and excellent value. Best coffee I've ever had. Highly recommended!"
    
    with col2:
        if st.button("🟡 Neutral Example", use_container_width=True):
            st.session_state.example_text = "The product arrived on time and was packaged well. It's an average snack, nothing special but not bad either."
    
    with col3:
        if st.button("🔴 Negative Example", use_container_width=True):
            st.session_state.example_text = "Terrible product! Arrived stale and tasted awful. Complete waste of money. Very disappointed with this purchase."
    
    # Text input
    default_text = st.session_state.get('example_text', '')
    review_text = st.text_area(
        "Enter your food review:",
        value=default_text,
        placeholder="Example: This coffee is amazing! Rich flavor and perfect aroma. Highly recommended!",
        height=150,
        key="review_input"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        analyze_button = st.button("🔍 Analyze", type="primary", use_container_width=True)
    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.example_text = ''
            st.rerun()
    
    if analyze_button and review_text.strip():
        with st.spinner("🤖 Analyzing sentiment with AI models..."):
            # Clean text
            cleaned_text = clean_text(review_text)
            
            # Get predictions
            vader_result = get_vader_prediction(cleaned_text, sia)
            roberta_result = get_roberta_prediction(cleaned_text, tokenizer, model, device)
            ensemble_result = get_ensemble_prediction(vader_result, roberta_result)
            
            # Display results
            st.markdown("---")
            st.subheader("📊 Analysis Results")
            
            # Overall sentiment cards
            col1, col2, col3 = st.columns(3)
            
            sentiment_colors = {
                'positive': '🟢',
                'neutral': '🟡',
                'negative': '🔴'
            }
            
            with col1:
                st.markdown(f"### VADER (17%)")
                st.markdown(f"## {sentiment_colors[vader_result['sentiment']]} {vader_result['sentiment'].upper()}")
                confidence = max(vader_result['negative'], vader_result['neutral'], vader_result['positive'])
                st.caption(f"Confidence: {confidence:.2%}")
            
            with col2:
                st.markdown(f"### RoBERTa (83%) ⭐")
                st.markdown(f"## {sentiment_colors[roberta_result['sentiment']]} {roberta_result['sentiment'].upper()}")
                confidence = max(roberta_result['negative'], roberta_result['neutral'], roberta_result['positive'])
                st.caption(f"Confidence: {confidence:.2%}")
            
            with col3:
                st.markdown(f"### Ensemble (84%)")
                st.markdown(f"## {sentiment_colors[ensemble_result['sentiment']]} {ensemble_result['sentiment'].upper()}")
                confidence = max(ensemble_result['negative'], ensemble_result['neutral'], ensemble_result['positive'])
                st.caption(f"Confidence: {confidence:.2%}")
            
            # Detailed scores
            st.markdown("---")
            st.subheader("📈 Detailed Sentiment Scores")
            
            # Create dataframe for comparison
            comparison_df = pd.DataFrame({
                'Model': ['VADER (17%)', 'RoBERTa Fine-tuned (83%)', 'Ensemble (84%)'],
                'Negative': [vader_result['negative'], roberta_result['negative'], ensemble_result['negative']],
                'Neutral': [vader_result['neutral'], roberta_result['neutral'], ensemble_result['neutral']],
                'Positive': [vader_result['positive'], roberta_result['positive'], ensemble_result['positive']]
            })
            
            # Bar chart
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Negative', x=comparison_df['Model'], y=comparison_df['Negative'], marker_color='#ff6b6b'))
            fig.add_trace(go.Bar(name='Neutral', x=comparison_df['Model'], y=comparison_df['Neutral'], marker_color='#ffd93d'))
            fig.add_trace(go.Bar(name='Positive', x=comparison_df['Model'], y=comparison_df['Positive'], marker_color='#6bcf7f'))
            
            fig.update_layout(
                barmode='group',
                title='Sentiment Scores Comparison',
                xaxis_title='Model',
                yaxis_title='Score',
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Performance insight
            st.info(f"""
            **💡 Best Prediction:** The **RoBERTa fine-tuned model** ({roberta_result['sentiment']}) is recommended 
            as it has 83% test accuracy, significantly outperforming VADER (17%).
            """)
            
            # Save to history
            st.session_state.history.append({
                'timestamp': pd.Timestamp.now(),
                'review': review_text[:100] + '...' if len(review_text) > 100 else review_text,
                'vader': vader_result['sentiment'],
                'roberta': roberta_result['sentiment'],
                'ensemble': ensemble_result['sentiment']
            })
            
    elif analyze_button:
        st.warning("⚠️ Please enter a review to analyze.")

def batch_analysis_mode(sia, tokenizer, model, device):
    """Batch analysis interface"""
    st.header("📊 Batch Analysis")
    
    st.markdown("""
    Upload a CSV file with a column named **'Text'** containing reviews.
    The app will analyze all reviews using the fine-tuned RoBERTa model.
    """)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        if 'Text' not in df.columns:
            st.error("❌ CSV file must contain a 'Text' column with reviews.")
            return
        
        st.success(f"✅ Loaded {len(df)} reviews")
        
        # Limit for performance
        max_reviews = 500
        if len(df) > max_reviews:
            st.warning(f"⚠️ Analyzing first {max_reviews} reviews for performance.")
            df = df.head(max_reviews)
        
        if st.button("🔍 Analyze All Reviews", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results = []
            
            for idx, row in df.iterrows():
                status_text.text(f"Analyzing review {idx + 1}/{len(df)}...")
                progress_bar.progress((idx + 1) / len(df))
                
                text = clean_text(row['Text'])
                
                vader_result = get_vader_prediction(text, sia)
                roberta_result = get_roberta_prediction(text, tokenizer, model, device)
                ensemble_result = get_ensemble_prediction(vader_result, roberta_result)
                
                results.append({
                    'Review': row['Text'][:100] + '...' if len(row['Text']) > 100 else row['Text'],
                    'VADER': vader_result['sentiment'],
                    'RoBERTa (Fine-tuned)': roberta_result['sentiment'],
                    'Ensemble': ensemble_result['sentiment'],
                    'VADER_Confidence': max(vader_result['negative'], vader_result['neutral'], vader_result['positive']),
                    'RoBERTa_Confidence': max(roberta_result['negative'], roberta_result['neutral'], roberta_result['positive']),
                    'Ensemble_Confidence': max(ensemble_result['negative'], ensemble_result['neutral'], ensemble_result['positive'])
                })
            
            status_text.text("✅ Analysis complete!")
            progress_bar.empty()
            
            results_df = pd.DataFrame(results)
            
            # Display summary statistics
            st.markdown("---")
            st.subheader("📈 Summary Statistics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### VADER (17%)")
                vader_counts = results_df['VADER'].value_counts()
                for sentiment in ['positive', 'neutral', 'negative']:
                    count = vader_counts.get(sentiment, 0)
                    st.metric(sentiment.capitalize(), f"{count} ({count/len(results_df)*100:.1f}%)")
            
            with col2:
                st.markdown("### RoBERTa (83%) ⭐")
                roberta_counts = results_df['RoBERTa (Fine-tuned)'].value_counts()
                for sentiment in ['positive', 'neutral', 'negative']:
                    count = roberta_counts.get(sentiment, 0)
                    st.metric(sentiment.capitalize(), f"{count} ({count/len(results_df)*100:.1f}%)")
            
            with col3:
                st.markdown("### Ensemble (84%)")
                ensemble_counts = results_df['Ensemble'].value_counts()
                for sentiment in ['positive', 'neutral', 'negative']:
                    count = ensemble_counts.get(sentiment, 0)
                    st.metric(sentiment.capitalize(), f"{count} ({count/len(results_df)*100:.1f}%)")
            
            # Pie charts
            st.markdown("---")
            st.subheader("📊 Sentiment Distribution")
            col1, col2, col3 = st.columns(3)
            
            colors = {'positive': '#6bcf7f', 'neutral': '#ffd93d', 'negative': '#ff6b6b'}
            
            with col1:
                fig = px.pie(values=vader_counts.values, names=vader_counts.index, 
                           title='VADER', color=vader_counts.index,
                           color_discrete_map=colors)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.pie(values=roberta_counts.values, names=roberta_counts.index,
                           title='RoBERTa Fine-tuned ⭐', color=roberta_counts.index,
                           color_discrete_map=colors)
                st.plotly_chart(fig, use_container_width=True)
            
            with col3:
                fig = px.pie(values=ensemble_counts.values, names=ensemble_counts.index,
                           title='Ensemble', color=ensemble_counts.index,
                           color_discrete_map=colors)
                st.plotly_chart(fig, use_container_width=True)
            
            # Show detailed results
            st.markdown("---")
            st.subheader("📋 Detailed Results")
            st.dataframe(results_df, use_container_width=True, height=400)
            
            # Download button
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="sentiment_analysis_results.csv",
                mime="text/csv"
            )

def view_history_mode():
    """View analysis history"""
    st.header("📜 Analysis History")
    
    if not st.session_state.history:
        st.info("No analysis history yet. Analyze some reviews to see them here!")
        return
    
    history_df = pd.DataFrame(st.session_state.history)
    
    st.markdown(f"### Total Analyses: {len(history_df)}")
    
    # Display history table
    st.dataframe(history_df, use_container_width=True, height=400)
    
    # Clear history button
    if st.button("🗑️ Clear History", type="secondary"):
        st.session_state.history = []
        st.rerun()
    
    # Download history
    csv = history_df.to_csv(index=False)
    st.download_button(
        label="📥 Download History as CSV",
        data=csv,
        file_name="analysis_history.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()