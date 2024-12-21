import time
import threading
from queue import PriorityQueue
from newspaper import Article, Source, Config
from threading import Lock
import nltk
import os
from concurrent.futures import ThreadPoolExecutor
from .scoring_engine import ScoringEngine
from .cache_manager import CacheManager
from urllib.parse import urlparse
from langdetect import detect
from datetime import datetime
import hashlib
import json

class ArticleManager:
    def __init__(self, sources, toplist_size=10, throttle_interval=2, auto_start=True, articles_per_source=5, scoring_weights_file='scoring_weights.json', cache_duration=3600, max_workers=None):
        """
        Initialize the ArticleManager.

        Args:
            sources (list): List of news source URLs.
            toplist_size (int): Number of articles to maintain in the toplist.
            throttle_interval (int): Minimum time (seconds) between requests to a website.
            auto_start (bool): Whether to automatically start the daemon threads.
            articles_per_source (int): Maximum number of articles to fetch per source.
            scoring_weights_file (str): Path to the JSON file containing scoring weights.
            cache_duration (int): Duration (seconds) to keep articles in cache.
            site_delay (int): Delay (seconds) between requests to the same site.
        """
        print(f"\n[INIT] Starting ArticleManager with {articles_per_source} articles per source")
        self._initialize_nltk()
        
        self.config = Config()
        self.config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        self.config.request_timeout = 10
        self.config.memoize_articles = False
        self.config.language = 'da'
        self.max_workers = max_workers
        
        self.sources = sources
        self.toplist_size = toplist_size
        self.throttle_interval = throttle_interval
        self.articles = []  # List to store articles
        self.toplist = []  # Initialize toplist

        self.prefetch_queue = PriorityQueue()
        self.download_queue = PriorityQueue()
        self.parse_queue = PriorityQueue()
        self.nlp_queue = PriorityQueue()

        self.daemon_running = False

        self.scoring_weights_file = scoring_weights_file
        self.scoring_weights = self._load_scoring_weights()

        self.last_access_times = {}
        self.article_counter = 0
        self.lock = Lock()

        self.articles_per_source = articles_per_source
        self.max_workers = self.max_workers or os.cpu_count()
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

        self.sleep_time = 0.01  # Short sleep time to release CPU

        self.scoring_engine = ScoringEngine()
        self.cache_manager = CacheManager(cache_duration=cache_duration)

        self.scoring_stats = {
            "total_scored": 0,
            "avg_score": 0.0,
            "last_update": None
        }

        self.content_weights = {
            "title": 0.4,
            "content": 0.6,
        }

        self.load_cached_articles()
        self.prefetch_articles()
        if auto_start:
            self.start_daemon()

        # Initialize last_distribute_time and current_distribution
        self.last_distribute_time = 0
        self.current_distribution = None

    def getArticleShortInfoTxt(text):
        last_char = -25
        if text:
            return f"Text: {text[last_char:]}"
        return "***"
    
    def getArticleShortInfo(self, article):
        last_char = -25
        retval = "No info in article"
        if article["title"]:
            retval = f"Title: {article['title'][last_char:]}"
        else:
            retval = f'url: {article["url"][last_char:]}'
        return retval
    
    def showCpuAndworkerForEachQueue(self):
        print(f"prefetch_queue: {self.prefetch_queue.qsize()} workers: {self.executor._max_workers}")
        print(f"download_queue: {self.download_queue.qsize()} workers: {self.executor._max_workers}")
        print(f"parse_queue: {self.parse_queue.qsize()} workers: {self.executor._max_workers}")
        print(f"nlp_queue: {self.nlp_queue.qsize()} workers: {self.executor._max_workers}")

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
        """Ensure requests to a source URL are throttled to respect the interval."""
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

    def _load_scoring_weights(self):
        """Load scoring weights from the JSON file."""
        if os.path.exists(self.scoring_weights_file):
            with open(self.scoring_weights_file, 'r', encoding='utf-8') as file:
                return json.load(file)
        return {}

    def _save_scoring_weights(self):
        """Save scoring weights to the JSON file."""
        with open(self.scoring_weights_file, 'w', encoding='utf-8') as file:
            json.dump(self.scoring_weights, file, indent=4)

    def load_cached_articles(self):
        """Load all cached minimalistic articles."""
        print("\n[CACHE] Loading cached articles")
        for source_url in self.sources:
            cached_article = self.cache_manager.load_from_cache(source_url)
            if cached_article:
                print(f"[CACHE] Loaded cached article: ", getArticleShortInfoTxt( 
                                                                   {cached_article['url']})
                    )
                self.toplist.append(cached_article)

    def prefetch_articles(self):
        """Fetch metadata (titles and URLs) for articles from sources."""
        print("\n[PREFETCH] Starting to prefetch articles")
        
        for source_url in self.sources:
            self._throttle_request(source_url)
            try:
                source = Source(source_url, config=self.config)
                print(f"[PREFETCH] Building source: ", self.getArticleShortInfo())
                source.build()
                
                if not source.articles:
                    print(f"[WARNING] No articles found for ", self.getArticleShortInfo())
                    continue
                
                sorted_articles = sorted(source.articles, key=lambda x: x.publish_date or 0, reverse=True)
                articles_to_process = sorted_articles[:self.articles_per_source]
                print(f"[INFO] Processing {len(articles_to_process)} of {len(source.articles)} articles from ", self.getArticleShortInfo())
                
                for article in articles_to_process:
                    if not article.url:
                        continue
                    print(f"[PREFETCH] Found article: ", self.getArticleShortInfo())
                    priority = self.get_next_priority()
                    article_obj = Article(article.url, config=self.config)
                    item = {
                        "url": article.url,
                        "source_url": source_url,
                        "article_obj": article_obj,
                        "title": None,
                        "score": None,
                        "content": None,
                        "summary": None
                    }
                    self.prefetch_queue.put((priority, item))
            except Exception as e:
                print(f"[ERROR] Building source ")
        
        if not self.daemon_running:
            self.start_daemon()

    def all_queues_empty(self):
        """Check if all queues are empty."""
        return (self.prefetch_queue.empty() and 
                self.download_queue.empty() and 
                self.parse_queue.empty() and 
                self.nlp_queue.empty())

    def distribute_workers(self):
        """
        Distribute workers across the queues (prefetch, download, parse, nlp)
        based on queue size—BUT only recalculate if at least 10 seconds have
        passed since the last distribution.
        """
        now = time.time()
        # Only recalculate if 10 seconds have passed
        if (now - self.last_distribute_time) < 10:
            # If we already have a distribution, just return it
            if self.current_distribution is not None:
                #print("Returning cached distribution (less than 10s since last calculation).")
                return self.current_distribution

        # Otherwise, we recalculate the distribution
        print("Recalculating worker distribution...")
        total_workers = self.max_workers
        min_workers = 1  # Each queue must have at least 1 worker

        # Get the current sizes of each queue
        queue_sizes = {
            'prefetch_queue': self.prefetch_queue.qsize(),
            'download_queue': self.download_queue.qsize(),
            'parse_queue': self.parse_queue.qsize(),
            'nlp_queue': self.nlp_queue.qsize()
        }

        # Initialize each queue with the minimum number of workers
        workers = {queue_name: min_workers for queue_name in queue_sizes}

        # Subtract the workers we have already assigned (1 each)
        total_workers -= len(queue_sizes) * min_workers

        # Distribute remaining workers based on the largest queue sizes first
        while total_workers > 0:
            # Find the queue with the maximum remaining size
            max_queue_name = max(queue_sizes, key=queue_sizes.get)

            # If that queue has no tasks waiting, break early
            if queue_sizes[max_queue_name] == 0:
                break

            # Assign one more worker to that queue
            workers[max_queue_name] += 1

            # Conceptually "decrement" a task from that queue
            queue_sizes[max_queue_name] -= 1

            # Decrement the total worker pool
            total_workers -= 1

        # Update time and store the new distribution
        self.last_distribute_time = now
        self.current_distribution = workers

        print(f"Worker distribution: {workers}")
        return workers

    def process_prefetch_queue(self):
        """Process articles from the prefetch queue and move to the download queue."""
        while self.daemon_running:
            workers = self.distribute_workers()
            self.executor._max_workers = workers['prefetch_queue']
            if self.prefetch_queue.empty():
                time.sleep(self.sleep_time)
                continue
                
            priority, article = self.prefetch_queue.get()
            print(f"[PREFETCH] Moving article to download queue: ", self.getArticleShortInfo())
            self.download_queue.put((priority, article))

    def process_download_queue(self):
        """Download article content and move to the parse queue."""
        while self.daemon_running:
            workers = self.distribute_workers()
            self.executor._max_workers = workers['download_queue']
            if self.download_queue.empty():
                time.sleep(self.sleep_time)
                continue
                
            priority, item = self.download_queue.get()
            article_obj = item["article_obj"]
            cached_article = self.cache_manager.load_from_cache(article_obj.url)
            if cached_article:
                print(f"[CACHE] Loaded article from cache: {article_obj.url}")
                self.toplist.append(cached_article)
                continue

            try:
                print(f"[DOWNLOAD] Processing: ", self.getArticleShortInfo())
                self._throttle_request(article_obj.url)
                self._apply_site_delay()  # Apply site-specific delay
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
            item["favicon"] = article_obj.meta_favicon or f"https://www.google.com/s2/favicons?domain={self._get_domain(item['url'])}" or "/static/default_favicon.ico"
            item["source_name"] = self._get_domain(item['url'])
            item["language"] = detect(article_obj.text) if article_obj.text else 'da'
            item["publish_date"] = article_obj.publish_date or datetime.now()
            
            if item["title"] and item["content"]:
                print(f"[PARSE] Successfully parsed: {item['url']}")
                self.nlp_queue.put((priority, item))
                return True
                print(f"[PARSE] Missing content after parse: {item['url']}")
                return False
                
        except Exception as e:
            print(f"[ERROR] Parsing article {item['url']}: {str(e)}")
            return False

    def process_parse_queue(self):
        """Parse downloaded articles concurrently and move to the NLP queue."""
        print(f"[PARSE] Starting parse queue processor with {self.max_workers} workers")
        while self.daemon_running:
            workers = self.distribute_workers()
            self.executor._max_workers = workers['parse_queue']
            try:
                if self.parse_queue.empty():
                    time.sleep(self.sleep_time)
                    continue

                items = []
                for _ in range(min(self.max_workers, self.parse_queue.qsize())):
                    if not self.parse_queue.empty():
                        items.append(self.parse_queue.get())
                
                if not items:
                    continue

                futures = []
                for priority, item in items:
                    future = self.executor.submit(self.process_single_parse, priority, item)
                    futures.append((future, priority, item))
                
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
            workers = self.distribute_workers()
            self.executor._max_workers = workers['nlp_queue']
            if self.nlp_queue.empty():
                time.sleep(self.sleep_time)
                continue

            priority, item = self.nlp_queue.get()
            try:

                # Start processing the article
                print(f"[NLP] Processing: {item['url']}")
                start_time = time.time()

                # Perform NLP on the article
                article_obj = item["article_obj"]
                nlp_start = time.time()
                article_obj.nlp()
                nlp_end = time.time()
                item["summary"] = article_obj.summary
                print(f"[NLP] Completed in {nlp_end - nlp_start:.2f} seconds")

                # Score the article's title
                print(f"[NLP SCORES] Starting")
                score_start = time.time()
                self.score_titles([item])
                score_end = time.time()
                print(f"[NLP SCORES] Completed in {score_end - score_start:.2f} seconds")
                print(f"[NLP Scored] {item['url']} with score {item['score']}")

                # Update the toplist with the new item
                print(f"[NLP Update toplist]")
                update_toplist_start = time.time()
                self.update_toplist(item)
                update_toplist_end = time.time()
                print(f"[NLP Updated toplist] Completed in {update_toplist_end - update_toplist_start:.2f} seconds")

                # Save the processed item to cache
                cache_start = time.time()
                self.cache_manager.save_to_cache(item)
                cache_end = time.time()
                print(f"[NLP] Saved to cache in {cache_end - cache_start:.2f} seconds")
                print(f"[NLP] Successfully processed: {item['url']} in {cache_end - start_time:.2f} seconds")

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
            
        interest_data = [(word, weight) for word, weight in self.scoring_weights.items()]
        
        for article in articles:
            title_score = self.scoring_engine.calculate_article_scores(
                [{"content": article["title"]}], 
                interest_data
            )[0] if article["title"] else 0
            
            content_score = self.scoring_engine.calculate_article_scores(
                [{"content": article["content"]}], 
                interest_data
            )[0] if article["content"] else 0
            
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
        """Add a new article to the toplist if it qualifies."""
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
        """Stop all daemon threads gracefully."""
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
            "favicon": article["favicon"] or "/static/default_favicon.ico",
            "source_name": self._get_domain(article["url"]),
            "publish_date": article["publish_date"].isoformat() if article.get("publish_date") else None
        }

    def get_toplist(self):
        """Retrieve the current toplist of articles."""
        sorted_toplist = sorted(self.toplist, key=lambda x: x["score"], reverse=True)
        return [self._serialize_article(article) for article in sorted_toplist]

    def get_minimalistic_article(self, article):
        """Extract minimalistic article object for rendering."""
        return {
            "url": article["url"],
            "title": article["title"],
            "summary": article["summary"],
            "score": article["score"],
            "favicon": article["favicon"] or "/static/default_favicon.ico",
            "source_name": self._get_domain(article["url"]),
            "publish_date": article["publish_date"].isoformat() if article.get("publish_date") else None
        }

    def get_minimalistic_toplist(self):
        """Retrieve the current toplist of articles in minimalistic form."""
        sorted_toplist = sorted(self.toplist, key=lambda x: x["score"], reverse=True)
        return [self.get_minimalistic_article(article) for article in sorted_toplist]

    def add_scoring_word(self, word, weight):
        """Dynamically add or update a scoring word and its weight."""
        self.scoring_weights[word.lower()] = weight
        self._save_scoring_weights()
        self.recalculate_scores()

    def remove_scoring_word(self, word):
        """Remove a scoring word from the scoring weights."""
        self.scoring_weights.pop(word.lower(), None)
        self._save_scoring_weights()
        self.recalculate_scores()

    def edit_scoring_word(self, word, new_weight):
        """Edit an existing scoring word with a new weight."""
        if word.lower() in self.scoring_weights:
            self.scoring_weights[word.lower()] = new_weight
            self._save_scoring_weights()
            self.recalculate_scores()
        else:
            print(f"Word '{word}' not found in scoring weights.")

    def recalculate_scores(self):
        """Recalculate scores for all articles based on updated scoring weights."""
        interest_data = [(word, weight) for word, weight in self.scoring_weights.items()]
        for article in self.toplist:
            title_score = self.scoring_engine.calculate_article_scores(
                [{"content": article["title"]}], 
                interest_data
            )[0] if article["title"] else 0
            
            content_score = self.scoring_engine.calculate_article_scores(
                [{"content": article["content"]}], 
                interest_data
            )[0] if article["content"] else 0
            
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
        self.toplist = sorted(self.toplist, key=lambda x: x["score"], reverse=True)[:self.toplist_size]

    def get_queue_status(self):
        """Retrieve the current status of all queues."""
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
            },
            "articles": [
                self.get_minimalistic_article(article)
                for article in self.toplist
            ]
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
