import os
import json
import time
from datetime import datetime
import hashlib

class CacheManager:
    def __init__(self, cache_dir='article_cache', cache_duration=3600):
        """
        Initialize the CacheManager.

        Args:
            cache_dir (str): Directory to store cached articles.
            cache_duration (int): Duration (seconds) to keep articles in cache.
        """
        self.cache_dir = cache_dir
        self.cache_duration = cache_duration
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_path(self, url):
        """Generate the cache file path for a given URL."""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{url_hash}.json")

    def save_to_cache(self, article_data):
        """Save minimalistic article data to cache."""
        minimalistic_article = {
            "url": article_data["url"],
            "title": article_data["title"],
            "summary": article_data["summary"],
            "score": article_data["score"],
            "favicon": article_data["favicon"],
            "source_name": article_data["source_name"],
            "publish_date": article_data["publish_date"].isoformat() if isinstance(article_data["publish_date"], datetime) else article_data["publish_date"],
            "content": article_data["content"],
            "score_details": article_data["score_details"]
        }
        cache_path = self._get_cache_path(article_data['url'])
        with open(cache_path, 'w') as cache_file:
            json.dump({'data': minimalistic_article, 'timestamp': time.time()}, cache_file)

    def load_from_cache(self, url):
        """Load minimalistic article data from cache."""
        cache_path = self._get_cache_path(url)
        if not os.path.exists(cache_path):
            return None

        try:
            with open(cache_path, 'r') as cache_file:
                cached_data = json.load(cache_file)
                if time.time() - cached_data['timestamp'] < self.cache_duration:
                    cached_data['data']['publish_date'] = datetime.fromisoformat(cached_data['data']['publish_date'])
                    return cached_data['data']
                else:
                    os.remove(cache_path)
        except (json.JSONDecodeError, OSError):
            if os.path.exists(cache_path):
                os.remove(cache_path)
        return None

    def clear_cache(self):
        """Clear all expired cache files."""
        for cache_file in os.listdir(self.cache_dir):
            cache_path = os.path.join(self.cache_dir, cache_file)
            try:
                with open(cache_path, 'r') as file:
                    cached_data = json.load(file)
                    if time.time() - cached_data['timestamp'] >= self.cache_duration:
                        os.remove(cache_path)
            except (json.JSONDecodeError, OSError):
                if os.path.exists(cache_path):
                    os.remove(cache_path)
