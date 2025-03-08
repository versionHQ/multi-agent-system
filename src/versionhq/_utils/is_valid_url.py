from urllib.parse import urlparse


def is_valid_url(url: str) -> bool:
    try:
        result = urlparse(url)
        return all(
            [
                result.scheme in ("http", "https", "gs"),
                result.netloc,
                len(result.netloc.split(".")) >= 2,  # Ensure domain has TLD
            ]
        )
    except Exception:
        return False
