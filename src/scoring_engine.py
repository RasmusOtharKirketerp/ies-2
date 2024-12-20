import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict, Tuple
import nltk
from nltk.stem import WordNetLemmatizer

class ScoringEngine:
    def __init__(self):
        """Initialize the scoring engine with language-specific resources."""
        self.vectorizers = {}
        self.lemmatizers = {}
        self.stopwords = {}
        self.translation_dict = self._initialize_translation_dict()
        self._initialize_nlp_resources()

    def _initialize_nlp_resources(self):
        """Initialize NLP resources for supported languages."""
        # Initialize for Danish
        self.lemmatizers['da'] = WordNetLemmatizer()  # Default to English for now
        self.stopwords['da'] = set(nltk.corpus.stopwords.words('danish'))
        
        # Initialize for English (fallback)
        self.lemmatizers['en'] = WordNetLemmatizer()
        self.stopwords['en'] = set(nltk.corpus.stopwords.words('english'))

    def _initialize_translation_dict(self) -> Dict[str, Dict[str, str]]:
        """Initialize a dictionary for translations."""
        return {
            'en': {
                'danmark': 'denmark',
                'teater': 'theater',
                'ukraine': 'ukraine',
                'rusland': 'russia',
                'krig': 'war',
                'fred': 'peace',
                'klima': 'climate',
                'teknologi': 'technology',
                'sundhed': 'health',
                'bitcoins': 'bitcoins',
                'lgbt': 'lgbt',
                'europa': 'europe',
                'zelenskyj': 'zelensky',
                'politi': 'police',
                'eksplosion': 'explosion',
                'kursfald': 'price drop',
                'københavn': 'copenhagen',
                'kina': 'china'
            },
            'da': {
                'denmark': 'danmark',
                'theater': 'teater',
                'ukraine': 'ukraine',
                'russia': 'rusland',
                'war': 'krig',
                'peace': 'fred',
                'climate': 'klima',
                'technology': 'teknologi',
                'health': 'sundhed',
                'bitcoins': 'bitcoins',
                'lgbt': 'lgbt',
                'europe': 'europa',
                'zelensky': 'zelenskyj',
                'police': 'politi',
                'explosion': 'eksplosion',
                'price drop': 'kursfald',
                'copenhagen': 'københavn',
                'china': 'kina'
            }
        }

    def calculate_article_scores(self, 
                                 articles: List[dict], 
                                 interest_data: List[Tuple[str, float]]) -> np.ndarray:
        """
        Calculate relevance scores for articles using vector space model.
        
        Args:
            articles: List of article dictionaries
            interest_data: List of (term, weight) tuples
        
        Returns:
            numpy array of scores
        """
        if not interest_data or not articles:
            print("No interest data or articles found")
            return np.zeros(len(articles))

        # Group articles by language
        articles_by_lang = self._group_by_language(articles)
        
        # Calculate scores for each language group
        final_scores = np.zeros(len(articles))
        
        for lang, group in articles_by_lang.items():
            try:
                translated_interest_data = self._translate_interest_data(interest_data, lang)
                lang_scores = self._calculate_language_scores(
                    group['articles'],
                    group['indices'],
                    translated_interest_data,
                    lang
                )
                
                # Map scores back to original indices
                for idx, score in zip(group['indices'], lang_scores):
                    final_scores[idx] = score
                    
            except Exception as e:
                print(f"Error processing {lang} articles: {e}")
                
        return final_scores

    def _group_by_language(self, articles: List[dict]) -> Dict:
        """Group articles by language."""
        groups = {}
        for i, article in enumerate(articles):
            lang = article.get('language', 'da')  # Default to Danish
            if lang not in groups:
                groups[lang] = {'articles': [], 'indices': []}
            groups[lang]['articles'].append(article)
            groups[lang]['indices'].append(i)
        return groups

    def _translate_interest_data(self, interest_data: List[Tuple[str, float]], target_lang: str) -> List[Tuple[str, float]]:
        """Translate interest data to the target language using a local dictionary."""
        translated_interest_data = []
        for term, weight in interest_data:
            translated_term = self.translation_dict.get(target_lang, {}).get(term, term)
            translated_interest_data.append((translated_term, weight))
        return translated_interest_data

    def _calculate_language_scores(self, 
                                   articles: List[dict], 
                                   indices: List[int],
                                   interest_data: List[Tuple[str, float]], 
                                   lang: str) -> np.ndarray:
        """Calculate scores for articles in a specific language."""
        # Prepare vectorizer
        if lang not in self.vectorizers:
            self.vectorizers[lang] = TfidfVectorizer(
                stop_words=list(self.stopwords.get(lang, [])),
                lowercase=True,
                token_pattern=r'\b\w+\b'
            )

        # Prepare texts and terms
        texts = [self._preprocess_text(article.get('content', ''), lang) 
                 for article in articles]
        
        interest_terms = [term for term, _ in interest_data]
        interest_weights = np.array([weight for _, weight in interest_data])
        
        # Calculate TF-IDF
        try:
            tfidf_matrix = self.vectorizers[lang].fit_transform(texts)
            
            # Calculate scores
            scores = np.zeros(len(texts))
            for term, weight in zip(interest_terms, interest_weights):
                term_vector = self.vectorizers[lang].transform([self._preprocess_text(term, lang)])
                term_scores = (tfidf_matrix * term_vector.T).toarray()
                scores += weight * np.squeeze(term_scores)
                
            return scores
            
        except Exception as e:
            print(f"Error in TF-IDF calculation: {e}")
            return np.zeros(len(texts))

    def _preprocess_text(self, text: str, lang: str) -> str:
        """Preprocess text for a specific language."""
        if not text:
            return ""
            
        # Lemmatize and clean text
        lemmatizer = self.lemmatizers.get(lang, self.lemmatizers['en'])
        words = text.lower().split()
        lemmatized = [lemmatizer.lemmatize(word) for word in words]
        return ' '.join(lemmatized)
