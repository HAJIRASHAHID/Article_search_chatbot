from newspaper import Article as NewsArticle


def fetch_full_content(url: str) -> str:
    try:
        article = NewsArticle(url)
        article.download()
        article.parse()
        return article.text or ""
    except Exception:
        return ""