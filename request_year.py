import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Set up Spotipy client credentials
client_id = "4f24e3e9bdbf40ffbda7496038221e13"
client_secret = "f9ec8c5073ec422d8910b1be593657a8"
client_credentials_manager = SpotifyClientCredentials(client_id, client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Function to get release year of a song given its name
def get_song_release_year(song_name):
    # Search for the song
    result = sp.search(q=song_name, type="track", limit=1)

    # Check if a track was found
    if len(result["tracks"]["items"]) == 0:
        print("No track found with that name.")
        return None

    # Get the release year from the album information
    album_id = result["tracks"]["items"][0]["album"]["id"]
    album = sp.album(album_id)
    release_year = album["release_date"][:4]

    return release_year

if __name__ == '__main__':
    # Example usage
    song_name = input("Enter song name: ")
    release_year = get_song_release_year(song_name)

    if release_year:
        print(f"{song_name} was released in {release_year}.")
