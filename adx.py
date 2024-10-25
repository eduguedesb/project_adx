import tweepy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Baixa os recursos de análise de sentimentos
nltk.download('vader_lexicon')

# Credenciais da API
api_key = 'SUA_API_KEY'
api_key_secret = 'SUA_API_SECRET_KEY'
access_token = 'SEU_ACCESS_TOKEN'
access_token_secret = 'SEU_ACCESS_SECRET_TOKEN'

# Autenticação com Tweepy
auth = tweepy.OAuthHandler(api_key, api_key_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# Função para coletar tweets com uma hashtag específica
def coletar_tweets(hashtag, quantidade=100):
    tweets_data = []
    for tweet in tweepy.Cursor(api.search_tweets, q=hashtag, lang="pt", tweet_mode='extended').items(quantidade):
        tweets_data.append(tweet.full_text)
    return tweets_data

# Coleta de tweets sobre um tópico específico
hashtag = "#saude"
tweets = coletar_tweets(hashtag, quantidade=200)

# Análise de Sentimentos
sia = SentimentIntensityAnalyzer()
sentimentos = [sia.polarity_scores(tweet)["compound"] for tweet in tweets]
sentimentos_df = pd.DataFrame({"Tweet": tweets, "Sentimento": sentimentos})

# Classifica os sentimentos
sentimentos_df['Classificação'] = sentimentos_df['Sentimento'].apply(lambda x: 'Positivo' if x > 0 else ('Negativo' if x < 0 else 'Neutro'))

# Visualização dos dados
plt.figure(figsize=(10, 6))
sns.countplot(data=sentimentos_df, x='Classificação', palette="viridis")
plt.title(f"Análise de Sentimentos para '{hashtag}'")
plt.xlabel("Classificação")
plt.ylabel("Número de Tweets")
plt.show()

# Palavras mais comuns
from collections import Counter
import re

# Limpeza de texto
def limpar_texto(texto):
    texto = re.sub(r'http\S+|www\S+|https\S+', '', texto, flags=re.MULTILINE)
    texto = re.sub(r'\@w+|\#', '', texto)
    texto = re.sub(r'[^A-Za-z0-9 ]+', '', texto)
    return texto.lower()

# Contagem das palavras mais comuns
tweets_limpos = [limpar_texto(tweet) for tweet in tweets]
palavras = " ".join(tweets_limpos).split()
contagem_palavras = Counter(palavras)

# Visualização das palavras mais comuns
palavras_comuns = pd.DataFrame(contagem_palavras.most_common(10), columns=['Palavra', 'Frequência'])

plt.figure(figsize=(10, 6))
sns.barplot(data=palavras_comuns, x='Frequência', y='Palavra', palette="viridis")
plt.title(f"Palavras Mais Comuns nos Tweets sobre '{hashtag}'")
plt.xlabel("Frequência")
plt.ylabel("Palavra")
plt.show()
