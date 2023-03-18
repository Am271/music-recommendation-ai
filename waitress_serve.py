from waitress import serve
import simple_flask

print("Serving the app")
serve(simple_flask.app, host='0.0.0.0', port=8080)