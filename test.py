import streamlit as st
import PyPDF2
import pdfplumber
import openai
import os
from dotenv import load_dotenv
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
from streamlit_lottie import st_lottie
import requests
import pandas as pd
import re
from datetime import datetime, timedelta
from typing import List, Dict

# Load environment variables
load_dotenv()

class PDFProcessor:
    def __init__(self):
        self.text_content = ""
        self.metadata = {}
    
    def extract_text_pypdf2(self, pdf_file):
        """Extract text using PyPDF2"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"Error extracting text with PyPDF2: {str(e)}")
            return ""
    
    def extract_text_pdfplumber(self, pdf_file):
        """Extract text using pdfplumber (more accurate)"""
        try:
            text = ""
            with pdfplumber.open(pdf_file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text
        except Exception as e:
            st.error(f"Error extracting text with pdfplumber: {str(e)}")
            return ""
    
    def process_pdf(self, pdf_file):
        """Process PDF and extract content"""
        # Try pdfplumber first (more accurate)
        text = self.extract_text_pdfplumber(pdf_file)
        
        # Fallback to PyPDF2 if pdfplumber fails
        if not text.strip():
            text = self.extract_text_pypdf2(pdf_file)
        
        self.text_content = text
        return text

class ContentAnalyzer:
    SUBTOPIC_STOPWORDS = set(["introduction", "overview", "summary", "conclusion", "chapter", "section"])

    def __init__(self):
        # Download required NLTK data
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
            nltk.download('stopwords', quiet=True)
        except Exception as e:
            print(f"Warning: Could not download NLTK data: {e}")
    
    def clean_text(self, text):
        """Clean and preprocess text"""
        # Remove extra whitespace and special characters
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.strip()
    
    def extract_topics(self, text, num_topics=5):
        """Extract main topics using TF-IDF and clustering"""
        try:
            # Clean text
            cleaned_text = self.clean_text(text)
            
            # Split into sentences
            sentences = nltk.sent_tokenize(cleaned_text)
            
            if len(sentences) < 2:
                # Fallback to keyword frequency extraction when not enough sentences
                from collections import Counter
                words = [w.lower() for w in nltk.word_tokenize(cleaned_text) if w.isalpha()]
                stop_words = set(nltk.corpus.stopwords.words('english'))
                words = [w for w in words if w not in stop_words]
                common = [w for w, _ in Counter(words).most_common(num_topics)]
                return [w.title() for w in common] if common else ["General Content"]
            
            # TF-IDF Vectorization
            vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            tfidf_matrix = vectorizer.fit_transform(sentences)
            
            # K-means clustering
            num_clusters = min(num_topics, len(sentences))
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            clusters = kmeans.fit_predict(tfidf_matrix)
            
            # Extract representative topics
            feature_names = vectorizer.get_feature_names_out()
            topics = []
            
            for i in range(num_clusters):
                # Get cluster center
                center = kmeans.cluster_centers_[i]
                # Get top features
                top_indices = center.argsort()[-3:][::-1]
                topic_words = [feature_names[idx] for idx in top_indices]
                topics.append(" ".join(topic_words).title())
            
            # If topics are generic or insufficient, fallback to keyword frequency method
            meaningful_topics = [t for t in topics if t.lower() != "general content"]
            if not meaningful_topics:
                # Simple frequency-based keywords
                from collections import Counter
                words = [w.lower() for w in nltk.word_tokenize(cleaned_text) if w.isalpha()]
                stop_words = set(nltk.corpus.stopwords.words('english'))
                words = [w for w in words if w not in stop_words]
                common = [w for w, _ in Counter(words).most_common(num_topics)]
                meaningful_topics = [w.title() for w in common]
            return meaningful_topics if meaningful_topics else ["General Content"]
            
        except Exception as e:
            st.error(f"Error in topic extraction: {str(e)}")
            return ["General Content"]
    
    def extract_subtopics(self, text, main_topic: str, num_subtopics: int = 5):
        """Extract subtopics related to a main topic within the text."""
        try:
            # Filter sentences mentioning the main topic (case-insensitive)
            sentences = [s for s in nltk.sent_tokenize(text) if main_topic.lower() in s.lower()]
            if not sentences:
                sentences = nltk.sent_tokenize(text)
            subset_text = " ".join(sentences)
            cleaned = self.clean_text(subset_text)
            words = [w.lower() for w in nltk.word_tokenize(cleaned) if w.isalpha()]
            stop_words = set(nltk.corpus.stopwords.words('english')) | self.SUBTOPIC_STOPWORDS
            words = [w for w in words if w not in stop_words and w != main_topic.lower()]
            from collections import Counter
            common = [w for w, _ in Counter(words).most_common(num_subtopics)]
            return [w.title() for w in common] if common else [main_topic]
        except Exception:
            return [main_topic]

    def estimate_difficulty(self, text):
        """Estimate content difficulty based on text complexity"""
        try:
            sentences = nltk.sent_tokenize(text)
            words = nltk.word_tokenize(text)
            
            # Calculate metrics
            avg_sentence_length = len(words) / len(sentences) if sentences else 0
            unique_words = len(set(words))
            total_words = len(words)
            vocabulary_richness = unique_words / total_words if total_words > 0 else 0
            
            # Simple difficulty scoring
            difficulty_score = (avg_sentence_length * 0.4 + vocabulary_richness * 60) / 100
            
            if difficulty_score < 0.3:
                return "Beginner"
            elif difficulty_score < 0.6:
                return "Intermediate"
            else:
                return "Advanced"
                
        except Exception:
            return "Intermediate"

class StudyPlanGenerator:
    def __init__(self):
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if self.openai_api_key:
            openai.api_key = self.openai_api_key
    
    def generate_study_plan(self, content, topics, difficulty, study_duration_weeks=4):
        """Generate a personalized study plan"""
        
        # Create a basic study plan structure
        plan = {
            "overview": {
                "difficulty": difficulty,
                "estimated_duration": f"{study_duration_weeks} weeks",
                "main_topics": topics
            },
            "weekly_schedule": []
        }
        
        # Generate weekly breakdown
        topics_per_week = max(1, len(topics) // study_duration_weeks)
        
        analyzer = ContentAnalyzer()
        for week in range(1, study_duration_weeks + 1):
            start_topic_idx = (week - 1) * topics_per_week
            end_topic_idx = min(week * topics_per_week, len(topics))
            week_topics = topics[start_topic_idx:end_topic_idx]
            
            if not week_topics and week == study_duration_weeks:
                week_topics = ["Review and Practice"]
            
            # Determine subtopics for the weekly schedule
            subtopics_for_week = []
            for mt in week_topics:
                subtopics_for_week.extend(analyzer.extract_subtopics(content, mt, 5))
            if not subtopics_for_week:
                subtopics_for_week = week_topics

            week_plan = {
                "week": week,
                "topics": week_topics,
                "daily_schedule": self._generate_daily_schedule(subtopics_for_week, difficulty, topics),
                "goals": self._generate_weekly_goals(week_topics, week, study_duration_weeks),
                "assessment": self._generate_quiz(subtopics_for_week)
            }
            
            plan["weekly_schedule"].append(week_plan)
        
        # Use OpenAI for enhanced plan if API key is available
        if self.openai_api_key:
            try:
                enhanced_plan = self._enhance_with_ai(content, plan)
                return enhanced_plan
            except Exception as e:
                st.warning(f"AI enhancement failed: {str(e)}. Using basic plan.")
        
        return plan
    
    def _recommend_resources(self, subtopic):
        """Return a list of diverse, high-quality learning resources (sites & books) for a subtopic."""
        base_query = subtopic.replace(' ', '+')
        return [
            f"Wikipedia – {subtopic} (https://en.wikipedia.org/wiki/{base_query})",
            f"TutorialsPoint on {subtopic} (https://www.tutorialspoint.com/index.htm#search:{base_query})",
            f"MIT OpenCourseWare videos on {subtopic} (https://ocw.mit.edu/search/?q={base_query})",
            f"Google Books results for {subtopic} (https://books.google.com/books?q={base_query})",
            f"YouTube playlist: {subtopic} (https://www.youtube.com/results?search_query={base_query}+tutorial)"
        ]

    def _generate_daily_schedule(self, topics, difficulty, all_topics=None):
        """Generate daily study schedule"""
        study_time_map = {
            "Beginner": 30,
            "Intermediate": 45,
            "Advanced": 60
        }
        
        daily_minutes = study_time_map.get(difficulty, 45)
        
        # Remove generic placeholder entries
        topics = [t for t in topics if t.lower() != "general content"]
        all_topics = [t for t in (all_topics or []) if t.lower() != "general content"]
        
        # Fallback to all_topics if week-specific topics are empty after filtering
        if not topics and all_topics:
            topics = all_topics
        
        schedule = []
        for day in range(1, 8):  # 7 days
            if day <= 5:  # Weekdays
                if topics:
                    topic = topics[(day - 1) % len(topics)]
                    activity = f"Study {topic} ({daily_minutes} min)"
                    resources = self._recommend_resources(topic)
                else:
                    activity = f"General study ({daily_minutes} min)"
            elif day == 6:  # Saturday
                activity = "Review and practice problems"
            else:  # Sunday
                activity = "Rest day or light review"
            
            entry = {
                "day": day,
                "activity": activity
            }
            if 'resources' in locals():
                entry['resources'] = resources
            schedule.append(entry)
        
        return schedule
    
    def _generate_weekly_goals(self, topics, week, total_weeks):
        """Generate weekly learning goals"""
        if week == 1:
            return [f"Understand basic concepts of {topic}" for topic in topics[:2]]
        elif week == total_weeks:
            return ["Complete final review", "Take practice assessment", "Consolidate knowledge"]
        else:
            return [f"Master {topic}" for topic in topics[:2]]
    
    def _generate_quiz(self, topics, num_questions=5):
        """Generate a simple quiz (list of questions) covering given topics."""
        questions = []
        for t in topics:
            questions.append(f"What is {t}?")
            if len(questions) >= num_questions:
                break
        # Fill remaining with explain/compare if needed
        i = 0
        while len(questions) < num_questions and i < len(topics):
            questions.append(f"Give an example use-case of {topics[i]}.")
            i += 1
        return questions or ["Review the week's material and write a summary."]

    def _enhance_with_ai(self, content, basic_plan):
        """Enhance study plan using OpenAI"""
        try:
            # Prepare content summary for AI
            content_summary = content[:2000] if len(content) > 2000 else content
            
            prompt = f"""
            Based on the following educational content, enhance this study plan with specific learning objectives, 
            study techniques, and detailed activities:

            Content Summary: {content_summary}

            Basic Plan: {basic_plan}

            Please provide enhanced weekly goals, specific study activities, and learning techniques 
            appropriate for the content difficulty level.
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500,
                temperature=0.7
            )
            
            # For now, return the basic plan with AI suggestions as notes
            basic_plan["ai_suggestions"] = response.choices[0].message.content
            return basic_plan
            
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def main():
    st.set_page_config(
        page_title="AI Study Planner",
        page_icon="📚",
        layout="wide"
    )
    
    # Load external CSS
    css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # Inject blue-white gradient background
    st.markdown(
        """
        <style>
        .stApp {
            background: #F8F9FA;
            background-attachment: fixed;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("📚 AI Study Planner")
    st.markdown("Upload a PDF and get a personalized study plan powered by AI!")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    study_weeks = st.sidebar.slider("Study Duration (weeks)", 1, 12, 4)
    num_topics = st.sidebar.slider("Number of Topics to Extract", 3, 10, 5)
    
    # OpenAI API Key input
    api_key = st.sidebar.text_input("OpenAI API Key (optional)", type="password", 
                                   help="Enter your OpenAI API key for enhanced AI features")
    if api_key:
        os.environ['OPENAI_API_KEY'] = api_key
    
    # Main content area
    upload_tab, plan_tab = st.tabs(["📄 Upload PDF", "📅 Study Plan"])

    with upload_tab:
        st.header("📄 Upload PDF")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
        if uploaded_file is not None:
            st.success(f"Uploaded: {uploaded_file.name}")
            
            # Process PDF
            with st.spinner("Processing PDF..."):
                processor = PDFProcessor()
                text_content = processor.process_pdf(uploaded_file)
            
            if text_content:
                st.success("✅ PDF processed successfully!")
                
                # Show text preview
                with st.expander("📖 Text Preview"):
                    st.text_area("Extracted Text", text_content[:1000] + "..." if len(text_content) > 1000 else text_content, height=200)
                
                # Analyze content
                with st.spinner("Analyzing content..."):
                    analyzer = ContentAnalyzer()
                    topics = analyzer.extract_topics(text_content, num_topics)
                    difficulty = analyzer.estimate_difficulty(text_content)
                
                st.subheader("📊 Content Analysis")
                st.write(f"**Difficulty Level:** {difficulty}")
                st.write(f"**Main Topics:** {', '.join(topics)}")
                
    with plan_tab:
        if 'text_content' in locals() and text_content:
            st.header("📅 Generated Study Plan")
            
            # Generate study plan
            with st.spinner("Generating personalized study plan..."):
                planner = StudyPlanGenerator()
                study_plan = planner.generate_study_plan(text_content, topics, difficulty, study_weeks)
            
            # Display study plan
            st.subheader("📋 Plan Overview")
            st.write(f"**Duration:** {study_plan['overview']['estimated_duration']}")
            st.write(f"**Difficulty:** {study_plan['overview']['difficulty']}")
            
            # Weekly breakdown
            for week_plan in study_plan['weekly_schedule']:
                with st.expander(f"📅 Week {week_plan['week']}: {', '.join(week_plan['topics'])}"):
                    st.write("**Goals:**")
                    for goal in week_plan['goals']:
                        st.write(f"• {goal}")
                    
                    st.write("**Daily Schedule:**")
                    for day in week_plan['daily_schedule']:
                        day_name = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][day['day']-1]
                        st.write(f"• {day_name}: {day['activity']}")
                        if 'resources' in day:
                            for res in day['resources']:
                                st.markdown(f"&nbsp;&nbsp;&nbsp;🔗 {res}")
                    
                    st.write("**Assessment Quiz:**")
                    for q in week_plan['assessment']:
                        st.write(f"• {q}")
            
            # AI suggestions if available
            if 'ai_suggestions' in study_plan:
                with st.expander("🤖 AI Enhancement Suggestions"):
                    st.write(study_plan['ai_suggestions'])
            
            # Download study plan
            if st.button("📥 Download Study Plan"):
                plan_text = f"""
AI STUDY PLANNER - GENERATED PLAN
=================================

Duration: {study_plan['overview']['estimated_duration']}
Difficulty: {study_plan['overview']['difficulty']}
Topics: {', '.join(study_plan['overview']['main_topics'])}

WEEKLY BREAKDOWN:
"""
                for week_plan in study_plan['weekly_schedule']:
                    plan_text += f"\nWEEK {week_plan['week']}: {', '.join(week_plan['topics'])}\n"
                    plan_text += "Goals:\n"
                    for goal in week_plan['goals']:
                        plan_text += f"  • {goal}\n"
                    plan_text += "Daily Schedule:\n"
                    for day in week_plan['daily_schedule']:
                        day_name = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][day['day']-1]
                        plan_text += f"  • {day_name}: {day['activity']}\n"
                    plan_text += f"Assessment: {week_plan['assessment']}\n"
                
                st.download_button(
                    label="Download as Text File",
                    data=plan_text,
                    file_name=f"study_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
    
    # Footer
    st.markdown("---")
    st.markdown("💡 **Tip:** For best results, upload PDFs with clear, educational content. The AI will analyze the text and create a structured learning path tailored to the content difficulty and your available time.")

if __name__ == "__main__":
    main()