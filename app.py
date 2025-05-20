import sys
if sys.platform.startswith('win'):
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import streamlit as st
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import pandas as pd
import plotly.express as px 
import unicodedata
import plotly.graph_objects as go
import plotly.express as px 
from typing import List, Dict
import math

import sys
sys.stdout.reconfigure(encoding='utf-8')
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True )



st.set_page_config(
    page_title=" French Medical NER", 
    page_icon="🧬", 
    layout="centered"
)

MODEL_NAME = "abdel132/ner-drbert-quaero"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
    return pipeline(
        "ner", 
        model=model, 
        tokenizer=tokenizer, 
        aggregation_strategy="simple"
    )
ner_pipeline = load_model()

def create_entity_distribution(df: pd.DataFrame) -> go.Figure:
    entity_counts = df['entity_group'].value_counts()
    colors = px.colors.qualitative.Set3[:len(entity_counts)]
    
    fig = go.Figure(data=[go.Pie(
        labels=entity_counts.index,
        values=entity_counts.values,
        hole=0.4,
        marker_colors=colors,
        textinfo='label+percent',
        hovertemplate="Entity: %{label}<br>Count: %{value}<br>Percentage: %{percent}<extra></extra>"
    )])
    
    fig.update_layout(
        title={
            'text': 'Distribution of Entity Types',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        width=700,
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    return fig

def create_confidence_distribution(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=df['score'],
        nbinsx=20,
        name='Confidence',
        marker_color='#4CAF50',
        hovertemplate="Confidence: %{x:.2%}<br>Count: %{y}<extra></extra>"
    ))

def paginate(entities, items_per_page=12):
    """Split entities into pages."""
    return [entities[i:i + items_per_page] for i in range(0, len(entities), items_per_page)]

    fig.update_layout(
        title={
            'text': 'Distribution of Confidence Scores',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title="Confidence Score",
        yaxis_title="Number of Entities",
        bargap=0.1,
        width=700,
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            tickformat='.0%',
            gridcolor='rgba(128,128,128,0.2)',
            zerolinecolor='rgba(128,128,128,0.2)'
        ),
        yaxis=dict(
            gridcolor='rgba(128,128,128,0.2)',
            zerolinecolor='rgba(128,128,128,0.2)'
        )
    )
    return fig

st.markdown("""
<style>
.entity-card {
    background-color: #0e1117;
    border-radius: 8px;
    padding: 15px;
    margin: 10px 0;
    border-left: 4px solid #2e7d32;
}
.confidence-high { border-left-color: #2e7d32; }
.confidence-medium { border-left-color: #f9a825; }
.confidence-low { border-left-color: #c62828; }
</style>
""", unsafe_allow_html=True)


def create_entity_distribution(df: pd.DataFrame) -> go.Figure:
    entity_counts = df['entity_group'].value_counts()
    colors = px.colors.qualitative.Set3[:len(entity_counts)]
    
    fig = go.Figure(data=[go.Pie(
        labels=entity_counts.index,
        values=entity_counts.values,
        hole=0.4,
        marker_colors=colors,
        textinfo='label+percent',
        hovertemplate="Entity: %{label}<br>Count: %{value}<br>Percentage: %{percent}<extra></extra>"
    )])
    
    fig.update_layout(
        title={
            'text': 'Distribution of Entity Types',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        width=700,
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    return fig



# ─── UI layout ────────────────────────────────────────────────────────────────
st.title("🧬 French Medical NER with DrBERT")
st.write("Paste a French clinical sentence to extract medical entities.")

text_input = st.text_area(
    "📝 Input text", 
    height=200, 
    placeholder="Le patient présente une hypertension artérielle..."
)

if 'entities' not in st.session_state:
    st.session_state.entities = None

if st.button("🔍 Extract Entities"):
    if not text_input.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Processing..."):
            st.session_state.entities = ner_pipeline(text_input)

# Display results if entities exist in session state
if st.session_state.entities:
    df = pd.DataFrame(st.session_state.entities)
    
    # Display summary metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Entities", len(st.session_state.entities))
    with col2:
        avg_confidence = df['score'].mean()
        st.metric("Average Confidence", f"{avg_confidence:.2%}")

    # Visualization options
    viz_type = st.selectbox(
        "Select Visualization",
        ["Entity Cards", "Entity Distribution"]
    )

    if viz_type == "Entity Cards":
        # Calculate pagination
        items_per_page = 12
        pages = paginate(st.session_state.entities, items_per_page)
        total_pages = len(pages)
        
        # Add page selector
        if total_pages > 1:
            page_number = st.number_input("Page", min_value=1, max_value=total_pages, value=1) - 1
        else:
            page_number = 0
            
        # Display current page entities in rows of 3
        current_page = pages[page_number]
        
        # Add CSS for card grid
        st.markdown("""
        <style>
        .card-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1rem;
            margin-bottom: 1rem;
        }
        
        </style>
        """, unsafe_allow_html=True)
        
        # Display entities in grid
        for i in range(0, len(current_page), 3):
            # Create a row of 3 cards
            row_entities = current_page[i:i+3]
            cols = st.columns(3)
            
            for j, (col, ent) in enumerate(zip(cols, row_entities)):
                with col:
                    confidence_class = (
                        "confidence-high" if ent['score'] > 0.8
                        else "confidence-medium" if ent['score'] > 0.6
                        else "confidence-low"
                    )
                    st.markdown(f"""
                    <div class="entity-card {confidence_class}">
                        <h4>{ent['word']}</h4>
                        <p>Type: {ent['entity_group']}</p>
                        <p>Confidence: {ent['score']:.2%}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
        # Display pagination info
        st.markdown(f"Page {page_number + 1} of {total_pages}")

    elif viz_type == "Entity Distribution":
        try:
            fig = create_entity_distribution(df)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating Entity Distribution: {str(e)}")

    

    # Export option
    df['word'] = df['word'].apply(lambda x: unicodedata.normalize('NFKC', x))
    csv_buffer = df.to_csv(
        index=False,
        encoding='utf-8-sig',
        sep=';'
    ).encode('utf-8-sig')
    
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv_buffer,
        file_name="medical_entities.csv",
        mime="text/csv;charset=utf-8",
    )