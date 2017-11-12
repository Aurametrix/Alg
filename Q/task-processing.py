import dramatiq
import requests

@dramatiq.actor
def count_words(url):
   response = requests.get(url)
   count = len(response.text.split(" "))
   print(f"There are {count} words at {url!r}.")

# Synchronously count the words on example.com in the current process
count_words("http://example.com")

# or send the actor a message so that it may perform the count
# later, in a separate process.
count_words.send("http://example.com")
