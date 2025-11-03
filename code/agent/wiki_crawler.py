import requests
from bs4 import BeautifulSoup

query = "人工智能"
url = "https://zh.wikipedia.org/wiki/{}".format(query)
url = "https://zh.wikipedia.org/wiki/%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36"
}
print(url)
response = requests.get(url, headers=headers)
print(response)
with open("output.html", "w", encoding="utf-8") as f:
    f.write(response.text)
print("网页已保存到 output.html")
soup = BeautifulSoup(response.text, 'html.parser')
content = soup.find(id="mw-content-text")
with open("output.html", "w", encoding="utf-8") as f:
    f.write(str(content))