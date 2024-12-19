from src.flask_handler import FlaskRoutesHandler
from src.article_manager import ArticleManager
import os

def main():
    print("Initializing development environment...")
    
    # Ensure we're in the project root directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Test sources for quick development
    sources = [
        "https://nyheder.tv2.dk",
    ]

    print("Setting up ArticleManager...")
    article_manager = ArticleManager(
        sources=sources,
        toplist_size=5,
        throttle_interval=1
    )
    
    print("Starting daemon...")
    article_manager.start_daemon()
    
    print("Starting Flask development server...")
    flask_handler = FlaskRoutesHandler(article_manager)
    
    try:
        flask_handler.app.run(
            host="localhost",
            port=5000,
            debug=True,
            use_reloader=True
        )
    except KeyboardInterrupt:
        print("\nShutting down...")
        article_manager.stop_daemon()

if __name__ == "__main__":
    main()
