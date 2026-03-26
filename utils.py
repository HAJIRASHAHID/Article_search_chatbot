from newspaper import Article

def extract_full_content(url: str):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception:
        return ""