import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from transformers import pipeline
import datetime

# --- ÉTAPE 1 : Récupération des données boursières (yfinance) ---
ticker = "AAPL"
print(f"Récupération des données pour {ticker}...")

# Récupération des 3 derniers mois
stock_data = yf.download(ticker, period="3mo", interval="1d")

# Nettoyage simple (gestion des MultiIndex si nécessaire selon version yfinance)
if isinstance(stock_data.columns, pd.MultiIndex):
    stock_data.columns = stock_data.columns.get_level_values(0)

# --- ÉTAPE 2 : Analyse de Sentiment (FinBERT) ---
print("Chargement du modèle FinBERT...")
# On utilise un pipeline pré-configuré pour l'analyse de sentiment
nlp = pipeline("sentiment-analysis", model="ProsusAI/finbert")

# Liste fictive de 10 titres d'actualités financières (mélange positif/négatif)
headlines = [
    "Apple reports record quarterly revenue thanks to iPhone sales.",          # Positif
    "Supply chain issues could delay the new Mac launch.",                     # Négatif
    "Analysts downgrade Apple stock citing market saturation.",                # Négatif
    "Apple announces huge stock buyback program.",                             # Positif
    "New regulatory scrutiny on App Store fees in Europe.",                    # Négatif
    "Apple car project rumors spark investor interest.",                       # Positif
    "Tech sector faces headwinds as interest rates rise.",                     # Négatif
    "Apple Vision Pro reviews are mixed, raising adoption concerns.",          # Neutre/Négatif
    "Partnership with OpenAI boosts AI capabilities for Siri.",               # Positif
    "Apple maintains dividend payout despite lower services growth."           # Neutre
]

# On associe arbitrairement ces news aux 10 derniers jours de trading pour l'exemple
# Dans un cas réel, vous auriez la date réelle de publication de la news.
last_10_dates = stock_data.index[-10:]

print("Analyse des news en cours...")
sentiment_scores = []
dates_news = []

for i, headline in enumerate(headlines):
    result = nlp(headline)[0] # Retourne {'label': 'positive', 'score': 0.95}
    
    score = result['score']
    label = result['label']
    
    # Conversion du label en score numérique signé (-1 à 1)
    if label == 'negative':
        final_score = -score
    elif label == 'neutral':
        final_score = 0
    else: # positive
        final_score = score
        
    sentiment_scores.append(final_score)
    dates_news.append(last_10_dates[i])

# Création d'un DataFrame pour les sentiments
df_sentiment = pd.DataFrame({'Date': dates_news, 'Sentiment': sentiment_scores})
df_sentiment.set_index('Date', inplace=True)

# Fusion avec les données boursières (Left Join pour garder tout l'historique de prix)
df_final = stock_data.join(df_sentiment)
df_final['Sentiment'] = df_final['Sentiment'].fillna(0) # 0 pour les jours sans news

# --- ÉTAPE 3 : Visualisation Interactive (Plotly) ---

# Création d'une figure avec 2 sous-graphiques (Prix en haut, Sentiment en bas)
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                    vertical_spacing=0.05, 
                    row_heights=[0.7, 0.3],
                    subplot_titles=(f'Cours de {ticker}', 'Sentiment Moyen des News'))

# Graphique 1 : Chandelier (Prix)
fig.add_trace(go.Candlestick(x=df_final.index,
                             open=df_final['Open'],
                             high=df_final['High'],
                             low=df_final['Low'],
                             close=df_final['Close'],
                             name='Prix Action'), row=1, col=1)

# Graphique 2 : Barres de couleur (Sentiment)
colors = ['green' if s > 0 else 'red' if s < 0 else 'gray' for s in df_final['Sentiment']]
fig.add_trace(go.Bar(x=df_final.index, y=df_final['Sentiment'],
                     marker_color=colors,
                     name='Score Sentiment'), row=2, col=1)

fig.update_layout(title_text=f"Analyse Quant : {ticker} vs News Sentiment",
                  xaxis_rangeslider_visible=False,
                  template="plotly_dark")

# Au lieu d'ouvrir le navigateur, on sauvegarde le rapport
date_str = datetime.date.today().strftime("%Y-%m-%d")
filename = f"Rapport_Trading_{date_str}.html"
fig.write_html(filename)
print(f"Rapport généré : {filename}")