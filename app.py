from src.flask_handler import FlaskRoutesHandler
from src.article_manager import ArticleManager
from dotenv import load_dotenv
import os

def main():
    # Load environment variables
    load_dotenv()
    
    # Get configuration from environment
    sources = os.getenv('NEWS_SOURCES').split(',')
    toplist_size = int(os.getenv('TOPLIST_SIZE', 10))
    throttle_interval = int(os.getenv('THROTTLE_INTERVAL', 2))
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_PORT', 5000))

    # Initialize ArticleManager with configuration
    article_manager = ArticleManager(
        sources=sources,
        toplist_size=toplist_size,
        throttle_interval=throttle_interval
    )

    # Start the article processing daemon
    article_manager.start_daemon()

    # Initialize and run Flask application
    flask_handler = FlaskRoutesHandler(article_manager)
    
    try:
        # Start the Flask server
        flask_handler.run(host=host, port=port)
    except KeyboardInterrupt:
        # Graceful shutdown
        article_manager.stop_daemon()

if __name__ == "__main__":
    main()
