import time
import threading
from queue import PriorityQueue
from newspaper import Article, Source, Config
from threading import Lock
import nltk
import os
from concurrent.futures import ThreadPoolExecutor
from .scoring_engine import ScoringEngine
from urllib.parse import urlparse
from langdetect import detect

class ArticleManager:
    def __init__(self, sources, toplist_size=10, throttle_interval=2, auto_start=True, articles_per_source=5):
        """
        Initialize the ArticleManager.

        Args:
            sources (list): List of news source URLs.
            toplist_size (int): Number of articles to maintain in the toplist.
            throttle_interval (int): Minimum time (seconds) between requests to a website.
            auto_start (bool): Whether to automatically start the daemon threads.
            articles_per_source (int): Maximum number of articles to fetch per source.
        """
        print(f"\n[INIT] Starting ArticleManager with {articles_per_source} articles per source")
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
            "danmark": 0.7,
            "teater": 0.8,
            "ukraine": 0.9,
            "rusland": 0.3,
            "krig": 0.8,
            "fred": 0.7,
            "klima": 0.3,
            "teknologi": 0.9,
            "sundhed": 0.6,
            "bitcoins": 1.0,
            "lgbt": -1.0,
            "europa": 0.5,
            "zelenskyj": 1.0,
            "politi": 0.5,
            "eksplosion": 0.7,
            "kursfald": 0.6,
            "københavn": 0.1,
            "kina": 0.8
        }

        self.last_access_times = {}  # Track the last access time for each source
        self.article_counter = 0  # Add counter for unique priorities
        self.lock = Lock()  # Ensure thread-safe updates to last_access_times and counter

        self.articles_per_source = articles_per_source
        self.max_workers = min(4, articles_per_source * len(sources))  # Adjust workers based on article count
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.articles_by_source = {}  # Track articles per source

        self.sleep_time = 0.5  # Sleep time when queues are empty

        self.scoring_engine = ScoringEngine()

        self.scoring_stats = {
            "total_scored": 0,
            "avg_score": 0.0,
            "last_update": None
        }

        self.content_weights = {
            "title": 0.4,    # Title importance weight
            "content": 0.6,  # Content importance weight
        }

        # Initialize articles after setup
        self.prefetch_articles()
        if auto_start:
            self.start_daemon()
        
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
        self.articles_by_source = {source: [] for source in self.sources}  # Reset tracking
        
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
                
                # Sort articles by publication date (newest first)
                sorted_articles = sorted(source.articles, key=lambda x: x.publish_date or 0, reverse=True)
                
                # Limit number of articles per source
                articles_to_process = sorted_articles[:self.articles_per_source]
                print(f"[INFO] Processing {len(articles_to_process)} of {len(source.articles)} articles from {source_url}")
                self.sources_obj[source_url] = source
                
                # Queue limited number of articles from the source
                for article in articles_to_process:
                    if not article.url:
                        continue
                    print(f"[PREFETCH] Found article: {article.url}")
                    priority = self.get_next_priority()
                    # Create Article object with config
                    article_obj = Article(article.url, config=self.config)
                    item = {
                        "url": article.url,
                        "source_url": source_url,  # Add source tracking
                        "article_obj": article_obj,
                        "title": None,
                        "score": None,
                        "content": None,
                        "summary": None
                    }
                    self.prefetch_queue.put((priority, item))
                    self.articles_by_source[source_url].append(item)
            except Exception as e:
                print(f"[ERROR] Building source {source_url}: {str(e)}")
        
        # Start the daemon after prefetching is complete
        if not self.daemon_running:
            self.start_daemon()

    def all_queues_empty(self):
        """Check if all queues are empty."""
        return (self.prefetch_queue.empty() and 
                self.download_queue.empty() and 
                self.parse_queue.empty() and 
                self.nlp_queue.empty())

    def process_prefetch_queue(self):
        """Process articles from the prefetch queue and move to the download queue."""
        while self.daemon_running:
            if self.prefetch_queue.empty():
                time.sleep(self.sleep_time)
                continue
                
            priority, article = self.prefetch_queue.get()
            print(f"[PREFETCH] Moving article to download queue: {article['url']}")
            self.download_queue.put((priority, article))

    def process_download_queue(self):
        """Download article content and move to the parse queue."""
        while self.daemon_running:
            if self.download_queue.empty():
                time.sleep(self.sleep_time)
                continue
                
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

    def _get_domain(self, url):
        """Extract the domain from a URL."""
        parsed_url = urlparse(url)
        return parsed_url.netloc

    def process_single_parse(self, priority, item):
        """Process a single article parse operation."""
        try:
            print(f"[PARSE] Processing: {item['url']}")
            article_obj = item["article_obj"]
            if not article_obj.html:
                print(f"[PARSE] No HTML content for: {item['url']}")
                return False
                
            article_obj.parse()
            
            if not article_obj.is_parsed:
                print(f"[PARSE] Failed to parse: {item['url']}")
                return False

            item["title"] = article_obj.title
            item["content"] = article_obj.text
            item["favicon"] = article_obj.meta_favicon or f"https://www.google.com/s2/favicons?domain={self._get_domain(item['url'])}"
            item["source_name"] = self._get_domain(item['url'])
            item["language"] = detect(article_obj.text) if article_obj.text else 'da'  # Detect language or default to Danish
            
            if item["title"] and item["content"]:
                print(f"[PARSE] Successfully parsed: {item['url']}")
                self.nlp_queue.put((priority, item))
                return True
            else:
                print(f"[PARSE] Missing content after parse: {item['url']}")
                return False
                
        except Exception as e:
            print(f"[ERROR] Parsing article {item['url']}: {str(e)}")
            return False

    def process_parse_queue(self):
        """Parse downloaded articles concurrently and move to the NLP queue."""
        print(f"[PARSE] Starting parse queue processor with {self.max_workers} workers")
        while self.daemon_running:
            try:
                if self.parse_queue.empty():
                    time.sleep(self.sleep_time)
                    continue

                # Get items to process
                items = []
                for _ in range(min(self.max_workers, self.parse_queue.qsize())):
                    if not self.parse_queue.empty():
                        items.append(self.parse_queue.get())
                
                if not items:
                    continue

                # Process batch
                futures = []
                for priority, item in items:
                    future = self.executor.submit(self.process_single_parse, priority, item)
                    futures.append((future, priority, item))
                
                # Wait for batch to complete
                for future, priority, item in futures:
                    try:
                        success = future.result(timeout=30)
                        if not success:
                            print(f"[PARSE] Failed to process: {item['url']}")
                    except Exception as e:
                        print(f"[ERROR] Future execution failed for {item['url']}: {str(e)}")

            except Exception as e:
                print(f"[ERROR] Parse queue processor error: {str(e)}")
                time.sleep(self.sleep_time)

    def process_nlp_queue(self):
        """Run NLP on parsed articles and update toplist."""
        while self.daemon_running:
            if self.nlp_queue.empty():
                time.sleep(self.sleep_time)
                continue
                
            priority, item = self.nlp_queue.get()
            try:
                print(f"[NLP] Processing: {item['url']}")
                article_obj = item["article_obj"]
                article_obj.nlp()
                item["summary"] = article_obj.summary
                self.score_titles([item])
                self.update_toplist(item)
                print(f"[NLP] Successfully processed: {item['url']}")
            except Exception as e:
                print(f"[ERROR] NLP processing {item['url']}: {str(e)}")

    def update_scoring_stats(self, score):
        """Update running statistics for scoring."""
        with self.lock:
            self.scoring_stats["total_scored"] += 1
            self.scoring_stats["avg_score"] = (
                (self.scoring_stats["avg_score"] * (self.scoring_stats["total_scored"] - 1) + score) 
                / self.scoring_stats["total_scored"]
            )
            self.scoring_stats["last_update"] = time.strftime("%Y-%m-%d %H:%M:%S")

    def score_titles(self, articles):
        """Compute scores using vector space model for both title and content."""
        if not articles:
            return
            
        # Convert scoring weights to interest data format
        interest_data = [(word, weight) for word, weight in self.scoring_weights.items()]
        
        for article in articles:
            # Score title
            title_score = self.scoring_engine.calculate_article_scores(
                [{"content": article["title"]}], 
                interest_data
            )[0] if article["title"] else 0
            
            # Score content
            content_score = self.scoring_engine.calculate_article_scores(
                [{"content": article["content"]}], 
                interest_data
            )[0] if article["content"] else 0
            
            # Combine scores with weights
            final_score = (
                self.content_weights["title"] * title_score +
                self.content_weights["content"] * content_score
            )
            
            article["score"] = float(final_score)
            article["score_details"] = {
                "title_score": float(title_score),
                "content_score": float(content_score),
                "weights": self.content_weights
            }
            self.update_scoring_stats(final_score)

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
        """Start the daemon threads to process all queues."""
        if not self.daemon_running:
            print("[DAEMON] Starting article processing daemon...")
            self.daemon_running = True
            
            # Start all processing threads
            threads = [
                threading.Thread(target=self.process_prefetch_queue, daemon=True),
                threading.Thread(target=self.process_download_queue, daemon=True),
                threading.Thread(target=self.process_parse_queue, daemon=True),
                threading.Thread(target=self.process_nlp_queue, daemon=True)
            ]
            
            for thread in threads:
                thread.start()
                print(f"[DAEMON] Started thread: {thread.name}")
            
            print("[DAEMON] All processing threads started")

    def stop_daemon(self):
        """
        Stop all daemon threads gracefully."""
        if self.daemon_running:
            print("[DAEMON] Stopping article processing daemon...")
            self.daemon_running = False
            self.executor.shutdown(wait=True)
            print("[DAEMON] All threads stopped")

    def _serialize_article(self, article):
        """Convert article to JSON-serializable format."""
        return {
            "url": article["url"],
            "title": article["title"],
            "score": article["score"],
            "score_details": article.get("score_details", {}),
            "content": article["content"],
            "summary": article["summary"],
            "favicon": article["favicon"],
            "source_name": self._get_domain(article["url"])  # Ensure correct domain extraction
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
        status = {
            "queues": {
                "prefetch_queue": self.prefetch_queue.qsize(),
                "download_queue": self.download_queue.qsize(),
                "parse_queue": self.parse_queue.qsize(),
                "nlp_queue": self.nlp_queue.qsize(),
            },
            "processing": {
                "is_active": not self.all_queues_empty() and self.daemon_running,
                "daemon_running": self.daemon_running,
                "total_articles": sum([q.qsize() for q in [
                    self.prefetch_queue, 
                    self.download_queue, 
                    self.parse_queue, 
                    self.nlp_queue
                ]])
            },
            "scoring": {
                "active_weights": self.scoring_weights,
                "stats": self.scoring_stats,
                "scored_articles": len(self.toplist),
                "scoring_engine": "vector_space_model"
            }
        }
        return status

    def get_queue_contents(self, queue):
        """Safely get contents of a queue without removing items."""
        with self.lock:
            return list(queue.queue) if not queue.empty() else []

    def get_articles_in_queue_for_source(self, queue, source_url):
        """Count articles for a specific source in a queue."""
        queue_contents = self.get_queue_contents(queue)
        return sum(1 for _, item in queue_contents if item["source_url"] == source_url)

    def get_sources(self):
        """
        Get the list of news sources with active status.

        Returns:
            dict: Dictionary with sources and their status
        """
        sources_status = {}
        for source_url in self.sources:
            try:
                # Use cached source object if available
                source_obj = self.sources_obj.get(source_url)
                if not source_obj:
                    source_obj = Source(source_url, config=self.config)
                    source_obj.build()
                
                # Count articles in each queue for this source
                in_prefetch = self.get_articles_in_queue_for_source(self.prefetch_queue, source_url)
                in_download = self.get_articles_in_queue_for_source(self.download_queue, source_url)
                in_parse = self.get_articles_in_queue_for_source(self.parse_queue, source_url)
                in_nlp = self.get_articles_in_queue_for_source(self.nlp_queue, source_url)
                
                # Get total articles originally fetched for this source
                total_articles = len(self.articles_by_source.get(source_url, []))

                sources_status[source_url] = {
                    "url": source_url,
                    "status": "active",
                    "article_count": total_articles,
                    "articles_by_stage": {
                        "prefetch": in_prefetch,
                        "download": in_download,
                        "parse": in_parse,
                        "nlp": in_nlp
                    }
                }
            except Exception as e:
                sources_status[source_url] = {
                    "url": source_url,
                    "status": "error",
                    "error": str(e),
                    "article_count": 0,
                    "articles_by_stage": {
                        "prefetch": 0,
                        "download": 0,
                        "parse": 0,
                        "nlp": 0
                    }
                }
        return sources_status
