from src.flask_handler import FlaskRoutesHandler
from src.article_manager import ArticleManager

if __name__ == "__main__":
    sources = [
        "https://www.dr.dk/nyheder",
        'https://nyheder.tv2.dk/',
        "https://politiken.dk",
        "https://www.bt.dk"
    ]
    
    # Create manager and start daemon immediately
    manager = ArticleManager(
        sources=sources,
        toplist_size=100,
        throttle_interval=3,
        auto_start=True,  # Changed to True
        articles_per_source=25
    )
    
    app = FlaskRoutesHandler(manager)
    app.app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
