from galiboo import auth, music
auth.set_api_key("<your api key>")

# Let's get the moods, emotions, & other music analysis data
# that Galiboo's Music A.I. has extracted for Coldplay's "Viva la Vida"

viva_la_vida = music.get_track("5a3fc326d836490c18703e3f")

print viva_la_vida['analysis']
print viva_la_vida['analysis']['smart_tags']
