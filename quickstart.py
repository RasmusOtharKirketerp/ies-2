from src.flask_handler import FlaskRoutesHandler
from src.article_manager import ArticleManager

if __name__ == "__main__":
    # Test with multiple Danish news sources
    sources = [
        "https://www.dr.dk/nyheder",
        "https://politiken.dk",
        "https://www.bt.dk"
    ]
    
    # Create manager with testing configuration
    manager = ArticleManager(
        sources=sources,
        toplist_size=5,
        throttle_interval=2,
        auto_start=False,  # Don't start daemon automatically
        articles_per_source=3  # Limit articles for testing
    )
    
    # Create and run Flask app (daemon can be started via API)
    app = FlaskRoutesHandler(manager)
    app.app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
