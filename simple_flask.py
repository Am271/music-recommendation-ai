from flask import Flask, request
import recommendation as rec
import request_year as ry

app = Flask(__name__)

@app.route("/recommend")
def recommend_song():
    args = request.args
    input = args.get('song', None)
    if input is None:
        print('No song specified')
    else:
        print(input)
        # Function call here
        year = ry.get_song_release_year(input)
        if year:
            songs = rec.recommend_songs([{'name' : input, 'year' : int(year)}], rec.data)
        else:
            print('Song not found')
            songs = 'Song not found'
    return songs