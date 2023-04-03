import requests


def query_plain(text, url="http://localhost:8888/plain"):
    return requests.post(url, json={"text": text}).json()


if __name__ == "__main__":
    text = "Autophagy maintains tumour growth through circulating arginine."
    print(query_plain(text))
