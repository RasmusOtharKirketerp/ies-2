from flask import Flask, jsonify, request, render_template, send_from_directory
from src.article_manager import ArticleManager
import os

class FlaskRoutesHandler:
    def __init__(self, article_manager):
        """
        Initialize the FlaskRoutesHandler.

        Args:
            article_manager (ArticleManager): Instance of the ArticleManager class.
        """
        self.article_manager = article_manager
        self.app = Flask(__name__, 
                        template_folder='../templates', 
                        static_folder='../static')  # Set template and static folders
        self.setup_routes()

    def setup_routes(self):
        """
        Setup all Flask routes.
        """
        @self.app.route("/", methods=["GET"])
        def main_page():
            """Display the main page with articles and controls."""
            status = self.article_manager.get_queue_status()
            articles = self.article_manager.get_toplist()
            return render_template('main.html', status=status, articles=articles)

        @self.app.route("/status", methods=["GET"])
        def get_status():
            """Get the status of all queues."""
            status = self.article_manager.get_queue_status()
            return jsonify(status)

        @self.app.route("/toplist", methods=["GET"])
        def get_toplist():
            """Get the toplist of articles."""
            toplist = self.article_manager.get_toplist()
            return jsonify(toplist)

        @self.app.route("/scoring-word", methods=["POST"])
        def add_scoring_word():
            """Add a new scoring word."""
            data = request.json
            word = data.get("word")
            weight = data.get("weight")
            if not word or weight is None:
                return jsonify({"error": "Word and weight are required."}), 400

            self.article_manager.add_scoring_word(word, weight)
            return jsonify({"message": f"Scoring word '{word}' added/updated successfully."})

        @self.app.route("/scoring-word", methods=["PUT"])
        def edit_scoring_word():
            """Edit an existing scoring word."""
            data = request.json
            word = data.get("word")
            new_weight = data.get("new_weight")
            if not word or new_weight is None:
                return jsonify({"error": "Word and new weight are required."}), 400

            self.article_manager.edit_scoring_word(word, new_weight)
            return jsonify({"message": f"Scoring word '{word}' updated successfully."})

        @self.app.route("/scoring-word", methods=["DELETE"])
        def remove_scoring_word():
            """Remove a scoring word."""
            data = request.json
            word = data.get("word")
            if not word:
                return jsonify({"error": "Word is required."}), 400

            self.article_manager.remove_scoring_word(word)
            return jsonify({"message": f"Scoring word '{word}' removed successfully."})

        @self.app.route("/daemon/start", methods=["POST"])
        def start_daemon():
            """Start the article processing daemon."""
            self.article_manager.start_daemon()
            return jsonify({"message": "Daemon started successfully."})

        @self.app.route("/daemon/stop", methods=["POST"])
        def stop_daemon():
            """Stop the article processing daemon."""
            self.article_manager.stop_daemon()
            return jsonify({"message": "Daemon stopped successfully."})

        @self.app.route("/sources", methods=["GET"])
        def get_sources():
            """Get the list of news sources with their status."""
            sources = self.article_manager.get_sources()
            return jsonify({
                "total_sources": len(sources),
                "sources": sources
            })

        @self.app.route("/docs", methods=["GET"])
        def get_docs():
            """Display API documentation and testing interface."""
            docs_html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>News API Documentation</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; max-width: 1000px; margin: 0 auto; padding: 20px; }
                    .endpoint { margin-bottom: 30px; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
                    .method { color: #008CBA; font-weight: bold; }
                    .curl { background: #f4f4f4; padding: 10px; overflow-x: auto; margin: 10px 0; }
                    .test-button { background: #008CBA; color: white; border: none; padding: 8px 16px; cursor: pointer; border-radius: 4px; }
                    .test-button:hover { background: #006B8F; }
                    .result { margin-top: 10px; padding: 10px; background: #f8f8f8; display: none; }
                    .url { color: #0056b3; text-decoration: none; }
                    .url:hover { text-decoration: underline; }
                </style>
                <script>
                    async function testEndpoint(method, endpoint, body = null) {
                        const resultDiv = document.getElementById(endpoint + '-result');
                        resultDiv.style.display = 'block';
                        resultDiv.innerHTML = 'Loading...';
                        
                        try {
                            const options = {
                                method: method,
                                headers: {
                                    'Content-Type': 'application/json'
                                }
                            };
                            if (body) {
                                options.body = JSON.stringify(body);
                            }
                            
                            const response = await fetch(endpoint, options);
                            const data = await response.json();
                            resultDiv.innerHTML = `<pre>${JSON.stringify(data, null, 2)}</pre>`;
                        } catch (error) {
                            resultDiv.innerHTML = `Error: ${error.message}`;
                        }
                    }
                </script>
            </head>
            <body>
                <h1>News API Documentation</h1>

                <div class="endpoint">
                    <h3>Get Queue Status</h3>
                    <p><span class="method">GET</span> <a href="/status" class="url" target="_blank">/status</a></p>
                    <div class="curl">curl http://localhost:5000/status</div>
                    <button class="test-button" onclick="testEndpoint('GET', '/status')">Test Endpoint</button>
                    <div id="/status-result" class="result"></div>
                </div>

                <div class="endpoint">
                    <h3>Get Top Articles</h3>
                    <p><span class="method">GET</span> <a href="/toplist" class="url" target="_blank">/toplist</a></p>
                    <div class="curl">curl http://localhost:5000/toplist</div>
                    <button class="test-button" onclick="testEndpoint('GET', '/toplist')">Test Endpoint</button>
                    <div id="/toplist-result" class="result"></div>
                </div>

                <div class="endpoint">
                    <h3>Add Scoring Word</h3>
                    <p><span class="method">POST</span> <span class="url">/scoring-word</span></p>
                    <div class="curl">
                        curl -X POST http://localhost:5000/scoring-word -H "Content-Type: application/json" -d '{"word": "technology", "weight": 1.5}'
                    </div>
                    <button class="test-button" onclick="testEndpoint('POST', '/scoring-word', {word: 'technology', weight: 1.5})">Test Endpoint</button>
                    <div id="/scoring-word-result" class="result"></div>
                </div>

                <div class="endpoint">
                    <h3>Edit Scoring Word</h3>
                    <p><span class="method">PUT</span> <span class="url">/scoring-word</span></p>
                    <div class="curl">
                        curl -X PUT http://localhost:5000/scoring-word -H "Content-Type: application/json" -d '{"word": "technology", "new_weight": 2.0}'
                    </div>
                    <button class="test-button" onclick="testEndpoint('PUT', '/scoring-word', {word: 'technology', new_weight: 2.0})">Test Endpoint</button>
                    <div id="/scoring-word-put-result" class="result"></div>
                </div>

                <div class="endpoint">
                    <h3>Remove Scoring Word</h3>
                    <p><span class="method">DELETE</span> <span class="url">/scoring-word</span></p>
                    <div class="curl">
                        curl -X DELETE http://localhost:5000/scoring-word -H "Content-Type: application/json" -d '{"word": "technology"}'
                    </div>
                    <button class="test-button" onclick="testEndpoint('DELETE', '/scoring-word', {word: 'technology'})">Test Endpoint</button>
                    <div id="/scoring-word-delete-result" class="result"></div>
                </div>

                <div class="endpoint">
                    <h3>Start Daemon</h3>
                    <p><span class="method">POST</span> <span class="url">/daemon/start</span></p>
                    <div class="curl">
                        curl -X POST http://localhost:5000/daemon/start
                    </div>
                    <button class="test-button" onclick="testEndpoint('POST', '/daemon/start')">Test Endpoint</button>
                    <div id="/daemon/start-result" class="result"></div>
                </div>

                <div class="endpoint">
                    <h3>Stop Daemon</h3>
                    <p><span class="method">POST</span> <span class="url">/daemon/stop</span></p>
                    <div class="curl">
                        curl -X POST http://localhost:5000/daemon/stop
                    </div>
                    <button class="test-button" onclick="testEndpoint('POST', '/daemon/stop')">Test Endpoint</button>
                    <div id="/daemon/stop-result" class="result"></div>
                </div>

                <div class="endpoint">
                    <h3>Get News Sources</h3>
                    <p><span class="method">GET</span> <a href="/sources" class="url" target="_blank">/sources</a></p>
                    <p class="description">Returns list of configured news sources with their current status and article count.</p>
                    <div class="curl">curl http://localhost:5000/sources</div>
                    <button class="test-button" onclick="testEndpoint('GET', '/sources')">Test Endpoint</button>
                    <div id="/sources-result" class="result"></div>
                </div>

            </body>
            </html>
            """
            return docs_html

        @self.app.route('/favicon.ico')
        def favicon():
            return send_from_directory(
                os.path.join(self.app.root_path, 'static'),
                'favicon.ico', mimetype='image/vnd.microsoft.icon'
            )

    def run(self, host="0.0.0.0", port=5000):
        """
        Run the Flask app.

        Args:
            host (str): The hostname to listen on. Defaults to "0.0.0.0".
            port (int): The port of the webserver. Defaults to 5000.
        """
        self.app.run(host=host, port=port, debug=True)


if __name__ == "__main__":
    article_manager = ArticleManager(sources=["https://example.com"], toplist_size=10, throttle_interval=2)
    flask_handler = FlaskRoutesHandler(article_manager)
    flask_handler.run()
