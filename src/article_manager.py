import time
import threading
from queue import PriorityQueue
from newspaper import Article, Source, Config
from threading import Lock
import nltk
import os

class ArticleManager:
    def __init__(self, sources, toplist_size=10, throttle_interval=2, auto_start=True):
        """
        Initialize the ArticleManager.

        Args:
            sources (list): List of news source URLs.
            toplist_size (int): Number of articles to maintain in the toplist.
            throttle_interval (int): Minimum time (seconds) between requests to a website.
            auto_start (bool): Whether to automatically start the daemon threads.
        """
        print(f"\n[INIT] Starting ArticleManager with sources: {sources}")
        # Ensure NLTK resources are downloaded
        self._initialize_nltk()
        
        # Configure newspaper
        self.config = Config()
        self.config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        self.config.request_timeout = 10
        self.config.memoize_articles = False
        self.config.language = 'da'  # Set Danish as language
        
        self.sources = sources
        self.sources_obj = {}  # Store Source objects
        self.toplist_size = toplist_size
        self.throttle_interval = throttle_interval
        self.articles = []  # Prefetched articles (metadata only)
        self.toplist = []  # Fully fetched and scored articles

        # Separate queues for each pipeline stage
        self.prefetch_queue = PriorityQueue()  # Queue for prefetch stage
        self.download_queue = PriorityQueue()  # Queue for download stage
        self.parse_queue = PriorityQueue()     # Queue for parse stage
        self.nlp_queue = PriorityQueue()       # Queue for NLP stage

        self.daemon_running = False
        self.daemon_thread = None
        self.scoring_weights = {  # Words and their associated weights
            "technology": 1.0,
            "politics": 0.8,
            "sports": 0.5,
            "entertainment": 0.3,
            "breaking": 1.2
        }

        self.last_access_times = {}  # Track the last access time for each source
        self.article_counter = 0  # Add counter for unique priorities
        self.lock = Lock()  # Ensure thread-safe updates to last_access_times and counter

        # Initialize articles after setup
        self.prefetch_articles()
        
    @staticmethod
    def _initialize_nltk():
        """Ensure all required NLTK resources are downloaded."""
        nltk_data_dir = os.path.join(os.getcwd(), '.venv', 'nltk_data')
        os.makedirs(nltk_data_dir, exist_ok=True)
        nltk.data.path.append(nltk_data_dir)
        
        required_packages = [
            'punkt',
            'averaged_perceptron_tagger',
            'maxent_ne_chunker',
            'words',
            'stopwords'
        ]
        
        for package in required_packages:
            try:
                nltk.data.find(f'tokenizers/{package}')
            except LookupError:
                print(f"Downloading NLTK package: {package}")
                nltk.download(package, download_dir=nltk_data_dir)

    def _throttle_request(self, source_url):
        """
        Ensure requests to a source URL are throttled to respect the interval.

        Args:
            source_url (str): The URL of the source being accessed.
        """
        with self.lock:
            last_access = self.last_access_times.get(source_url, 0)
            elapsed_time = time.time() - last_access
            if elapsed_time < self.throttle_interval:
                time.sleep(self.throttle_interval - elapsed_time)
            self.last_access_times[source_url] = time.time()

    def get_next_priority(self):
        """Get next priority number thread-safely."""
        with self.lock:
            self.article_counter += 1
            return self.article_counter

    def prefetch_articles(self):
        """Fetch metadata (titles and URLs) for articles from sources."""
        print("\n[PREFETCH] Starting to prefetch articles")
        for source_url in self.sources:
            self._throttle_request(source_url)
            try:
                # Create and build Source object with config
                source = Source(source_url, config=self.config)
                print(f"[PREFETCH] Building source: {source_url}")
                source.build()
                
                if not source.articles:
                    print(f"[WARNING] No articles found for {source_url}")
                    continue
                
                print(f"[INFO] Found {len(source.articles)} articles from {source_url}")
                self.sources_obj[source_url] = source
                
                # Queue each article from the source
                for article in source.articles:
                    if not article.url:
                        continue
                    print(f"[PREFETCH] Found article: {article.url}")
                    priority = self.get_next_priority()
                    # Create Article object with config
                    article_obj = Article(article.url, config=self.config)
                    item = {
                        "url": article.url,
                        "article_obj": article_obj,
                        "title": None,
                        "score": None,
                        "content": None,
                        "summary": None
                    }
                    self.prefetch_queue.put((priority, item))
            except Exception as e:
                print(f"[ERROR] Building source {source_url}: {str(e)}")

    def process_prefetch_queue(self):
        """Process articles from the prefetch queue and move to the download queue."""
        while self.daemon_running:
            if not self.prefetch_queue.empty():
                priority, article = self.prefetch_queue.get()
                print(f"[PREFETCH] Moving article to download queue: {article['url']}")
                self.download_queue.put((priority, article))

    def process_download_queue(self):
        """Download article content and move to the parse queue."""
        while self.daemon_running:
            if not self.download_queue.empty():
                priority, item = self.download_queue.get()
                article_obj = item["article_obj"]
                try:
                    print(f"[DOWNLOAD] Processing: {article_obj.url}")
                    self._throttle_request(article_obj.url)
                    article_obj.download()
                    if article_obj.html:
                        self.parse_queue.put((priority, item))
                    else:
                        print(f"[ERROR] No HTML content for {article_obj.url}")
                except Exception as e:
                    print(f"[ERROR] Downloading article {article_obj.url}: {str(e)}")

    def process_parse_queue(self):
        """Parse downloaded articles and move to the NLP queue."""
        while self.daemon_running:
            if not self.parse_queue.empty():
                priority, item = self.parse_queue.get()
                article_obj = item["article_obj"]
                try:
                    print(f"[PARSE] Processing: {article_obj.url}")
                    article_obj.parse()
                    item["title"] = article_obj.title
                    item["content"] = article_obj.text
                    self.nlp_queue.put((priority, item))
                except Exception as e:
                    print(f"[ERROR] Parsing article: {e}")

    def process_nlp_queue(self):
        """Run NLP on parsed articles and update toplist."""
        while self.daemon_running:
            if not self.nlp_queue.empty():
                priority, item = self.nlp_queue.get()
                article_obj = item["article_obj"]
                try:
                    print(f"[NLP] Processing: {article_obj.url}")
                    article_obj.nlp()
                    item["summary"] = article_obj.summary
                    self.score_titles([item])
                    self.update_toplist(item)
                except Exception as e:
                    print(f"[ERROR] NLP processing: {e}")

    def score_titles(self, articles):
        """Compute scores for given article titles."""
        for article in articles:
            if article["title"]:
                score = sum(self.scoring_weights.get(word.lower(), 0) for word in article["title"].split())
                article["score"] = score

    def update_toplist(self, article):
        """
        Add a new article to the toplist if it qualifies.

        Args:
            article (dict): Fully fetched article with score.
        """
        if len(self.toplist) < self.toplist_size:
            self.toplist.append(article)
        else:
            self.toplist.sort(key=lambda x: x["score"])
            if article["score"] > self.toplist[0]["score"]:
                self.toplist[0] = article

    def start_daemon(self):
        """
        Start the daemon threads to process all queues."""
        if not self.daemon_running:
            print("[DAEMON] Starting article processing daemon...")
            self.daemon_running = True
            threading.Thread(target=self.process_prefetch_queue, daemon=True).start()
            threading.Thread(target=self.process_download_queue, daemon=True).start()
            threading.Thread(target=self.process_parse_queue, daemon=True).start()
            threading.Thread(target=self.process_nlp_queue, daemon=True).start()
            print("[DAEMON] All processing threads started")

    def stop_daemon(self):
        """
        Stop all daemon threads gracefully."""
        if self.daemon_running:
            print("[DAEMON] Stopping article processing daemon...")
            self.daemon_running = False
            print("[DAEMON] Stop signal sent to all threads")

    def _serialize_article(self, article):
        """Convert article to JSON-serializable format."""
        return {
            "url": article["url"],
            "title": article["title"],
            "score": article["score"],
            "content": article["content"],
            "summary": article["summary"]
        }

    def get_toplist(self):
        """
        Retrieve the current toplist of articles.

        Returns:
            list: Top `x` articles sorted by score.
        """
        sorted_toplist = sorted(self.toplist, key=lambda x: x["score"], reverse=True)
        return [self._serialize_article(article) for article in sorted_toplist]

    def add_scoring_word(self, word, weight):
        """
        Dynamically add or update a scoring word and its weight.

        Args:
            word (str): The word to be added or updated.
            weight (float): The weight associated with the word.
        """
        self.scoring_weights[word.lower()] = weight
        self.recalculate_scores()

    def remove_scoring_word(self, word):
        """
        Remove a scoring word from the scoring weights.

        Args:
            word (str): The word to be removed.
        """
        self.scoring_weights.pop(word.lower(), None)
        self.recalculate_scores()

    def edit_scoring_word(self, word, new_weight):
        """
        Edit an existing scoring word with a new weight.

        Args:
            word (str): The word to be updated.
            new_weight (float): The new weight associated with the word.
        """
        if word.lower() in self.scoring_weights:
            self.scoring_weights[word.lower()] = new_weight
            self.recalculate_scores()
        else:
            print(f"Word '{word}' not found in scoring weights.")

    def recalculate_scores(self):
        """
        Recalculate scores for all articles based on updated scoring weights.
        """
        for article in self.articles:
            if article.get("title"):
                score = sum(self.scoring_weights.get(word.lower(), 0) for word in article["title"].split())
                article["score"] = score
        self.toplist = sorted(self.articles, key=lambda x: x.get("score", 0), reverse=True)[:self.toplist_size]

    def get_queue_status(self):
        """
        Retrieve the current status of all queues.

        Returns:
            dict: A dictionary containing the size of each queue.
        """
        return {
            "prefetch_queue": self.prefetch_queue.qsize(),
            "download_queue": self.download_queue.qsize(),
            "parse_queue": self.parse_queue.qsize(),
            "nlp_queue": self.nlp_queue.qsize()
        }

    def get_sources(self):
        """
        Get the list of news sources with active status.

        Returns:
            dict: Dictionary with sources and their status
        """
        sources_status = {}
        for source in self.sources:
            try:
                source_obj = Source(source)
                sources_status[source] = {
                    "url": source,
                    "status": "active",
                    "article_count": len(source_obj.articles) if source_obj.articles else 0
                }
            except Exception as e:
                sources_status[source] = {
                    "url": source,
                    "status": "error",
                    "error": str(e)
                }
        return sources_status
