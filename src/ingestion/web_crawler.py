import hashlib
import re
import time
from collections import deque
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser

import httpx
from bs4 import BeautifulSoup

from config import settings

_UNSAFE_CHARS = re.compile(r'[\\/:*?"<>|\x00-\x1f]')
_INDEX_NAMES = {"index.html", "index.htm", "default.html"}
_NON_HTML_EXTENSIONS = {
    ".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx", ".csv", ".zip", ".rar", ".7z",
    ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".ico", ".bmp",
    ".css", ".js", ".json", ".xml", ".rss", ".mp3", ".mp4", ".avi", ".mov", ".wav",
}


def normalize_url(url: str, base_url: str | None = None) -> str:
    """Resolves a possibly-relative URL to an absolute, canonical form.

    Args:
        url (str): URL to normalize, absolute or relative.
        base_url (str | None): Page the URL was found on, used to resolve
            relative hrefs. None if `url` is already absolute.

    Returns:
        str: Absolute URL with no fragment, host lower-cased, and query
            string stripped if `settings.crawl_strip_query_strings`.
    """
    resolved = urljoin(base_url, url) if base_url else url
    parsed = urlparse(resolved)
    query = "" if settings.crawl_strip_query_strings else parsed.query
    return urlunparse(parsed._replace(netloc=parsed.netloc.lower(), fragment="", query=query))


def is_in_scope(url: str, root_urls: list[str]) -> bool:
    """Checks whether `url` is on the same host as, and nested under (or
    equal to), at least one of `root_urls` — each root defines its own
    independent scope, so crawling several at once (even across different
    hosts) never leaks into a root's parent or an unrelated section.

    Args:
        url (str): URL to check.
        root_urls (list[str]): Crawl roots defining the allowed scope(s).

    Returns:
        bool: True if `url` should be followed.
    """
    target = urlparse(url)
    if target.scheme not in ("http", "https"):
        return False
    for root_url in root_urls:
        root = urlparse(root_url)
        if target.netloc.lower() != root.netloc.lower():
            continue
        root_path = root.path.rstrip("/")
        if target.path == root_path or target.path.startswith(root_path + "/"):
            return True
    return False


def _sanitize_segment(segment: str) -> str:
    """Turns one URL path segment into a filesystem-safe directory name.

    Args:
        segment (str): Raw, percent-encoded URL path segment.

    Returns:
        str: Sanitized name; a short hash if sanitization empties it out.
    """
    decoded = unquote(segment)
    cleaned = re.sub(r"\s+", "_", _UNSAFE_CHARS.sub("_", decoded)).strip("._ ")
    return cleaned or hashlib.sha1(segment.encode("utf-8")).hexdigest()[:10]


def url_to_local_path(url: str, base_dir: Path) -> Path:
    """Maps a URL's path to a local file path, mirroring each path segment
    as a directory and always saving the page as `index.html`.

    Saving every page as `index.html` inside its own directory (rather than
    `<last-segment>.html`) avoids a filesystem collision when a URL is both
    a page and the parent of other pages (e.g. `.../Studiare` is a page, and
    `.../Studiare/Piano` is also a page).

    Args:
        url (str): Absolute URL (already normalized).
        base_dir (Path): Root directory the crawl writes into.

    Returns:
        Path: `base_dir/<segment>/.../index.html`.
    """
    segments = [s for s in urlparse(url).path.split("/") if s]
    if segments and segments[-1].lower() in _INDEX_NAMES:
        segments = segments[:-1]
    sanitized = [_sanitize_segment(s) for s in segments]
    return base_dir.joinpath(*sanitized, "index.html") if sanitized else base_dir / "index.html"


def _get_robot_parser(url: str, cache: dict[str, RobotFileParser]) -> RobotFileParser:
    """Fetches and caches (per-domain) the robots.txt rules for `url`'s host.

    Args:
        url (str): URL whose host's robots.txt is needed.
        cache (dict[str, RobotFileParser]): Per-crawl cache, mutated in place.

    Returns:
        RobotFileParser: Parser for that host (allow-all if unreachable).
    """
    domain = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
    if domain not in cache:
        rp = RobotFileParser()
        rp.set_url(f"{domain}/robots.txt")
        try:
            rp.read()
        except OSError:
            rp.allow_all = True
        cache[domain] = rp
    return cache[domain]


def can_fetch(url: str, cache: dict[str, RobotFileParser], user_agent: str) -> bool:
    """Checks robots.txt permission for fetching `url`.

    Args:
        url (str): URL to check.
        cache (dict[str, RobotFileParser]): Per-crawl robots.txt cache.
        user_agent (str): User-Agent identifying this crawler.

    Returns:
        bool: True if allowed.
    """
    return _get_robot_parser(url, cache).can_fetch(user_agent, url)


def extract_links(html: str, page_url: str) -> list[str]:
    """Extracts in-page `<a href>` links, resolved to absolute URLs.

    Args:
        html (str): Page HTML.
        page_url (str): URL the page was fetched from (for relative hrefs).

    Returns:
        list[str]: Absolute HTTP(S) URLs pointing at non-HTML-excluded
            targets, deduplicated, in document order.
    """
    soup = BeautifulSoup(html, "html.parser")
    seen: set[str] = set()
    links: list[str] = []
    for tag in soup.find_all("a", href=True):
        href = tag["href"].strip()
        if not href or href.startswith("#"):
            continue
        if urlparse(href).scheme not in ("", "http", "https"):
            continue  # mailto:, tel:, javascript:, ...
        absolute = normalize_url(href, base_url=page_url)
        if Path(urlparse(absolute).path).suffix.lower() in _NON_HTML_EXTENSIONS:
            continue
        if absolute not in seen:
            seen.add(absolute)
            links.append(absolute)
    return links


def url_sidecar_path(local_path: Path) -> Path:
    """Path of the plain-text file recording the exact URL a saved page came
    from, since `url_to_local_path`'s sanitization isn't perfectly reversible
    (accents, case, encoding) — the local folder structure alone can't be
    trusted to reconstruct the original address for citing back to students.

    Args:
        local_path (Path): Path a page was saved to (e.g. `.../index.html`).

    Returns:
        Path: Same path with `.url` appended (e.g. `.../index.html.url`).
    """
    return local_path.with_name(local_path.name + ".url")


def save_url_sidecar(local_path: Path, url: str) -> None:
    """Records the exact URL a saved page came from, next to it.

    Args:
        local_path (Path): Path the page was saved to.
        url (str): The page's final (post-redirect) URL.
    """
    url_sidecar_path(local_path).write_text(url, encoding="utf-8")


def fetch_and_save(
    url: str, output_dir: Path, client: httpx.Client, user_agent: str
) -> tuple[Path, str, str] | None:
    """Downloads one page and saves it to its URL-mirrored local path.

    Args:
        url (str): URL to fetch.
        output_dir (Path): Root directory to save under.
        client (httpx.Client): Shared HTTP client (redirects followed).
        user_agent (str): Sent as the User-Agent header.

    Returns:
        tuple[Path, str, str] | None: (local_path, html_text, final_url) on
            success, or None on request failure / non-2xx / non-HTML response.
    """
    try:
        response = client.get(url)
    except httpx.HTTPError:
        return None
    if not (200 <= response.status_code < 300):
        return None
    content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
    if content_type != "text/html":
        return None

    final_url = str(response.url)
    local_path = url_to_local_path(final_url, output_dir)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_text(response.text, encoding="utf-8")
    save_url_sidecar(local_path, final_url)
    return local_path, response.text, final_url


def crawl_site(
    root_urls: str | list[str],
    output_dir: Path,
    delay: float = settings.crawl_delay_seconds,
    user_agent: str = settings.crawl_user_agent,
    max_pages: int | None = settings.crawl_max_pages,
) -> dict[str, list[str]]:
    """Crawls every in-scope HTML page reachable from one or more root URLs,
    saving each one to a local path that mirrors its URL path. Roots can be
    on different hosts (or different sections of the same host) in a single
    run — each defines its own independent scope (`is_in_scope`) *and* its
    own `max_pages` budget, so a large section doesn't starve a smaller one
    queued after it.

    Args:
        root_urls (str | list[str]): Page(s) to start from; each also
            defines its own allowed scope (same host, same-or-nested path).
            A single URL is accepted as a plain string too.
        output_dir (Path): Root directory to save pages under.
        delay (float): Minimum seconds between network fetches.
        user_agent (str): Sent as the User-Agent header and used for
            robots.txt permission checks.
        max_pages (int | None): Safety cap on the number of network fetches,
            applied *per root* (cache hits from a prior run don't count
            against it) — not a total shared across all of them.

    Returns:
        dict[str, list[str]]: {"failures": [...], "skipped_robots": [...]}
            URLs that errored, and URLs disallowed by robots.txt.
    """
    if isinstance(root_urls, str):
        root_urls = [root_urls]

    normalized_roots = []
    for root_url in root_urls:
        if not urlparse(root_url).scheme:
            root_url = f"https://{root_url}"
        normalized_roots.append(normalize_url(root_url))

    def root_index_for(url: str) -> int | None:
        """Index of the first root `url` falls under, or None (shouldn't
        happen for a URL that was enqueued via `is_in_scope`)."""
        for i, root in enumerate(normalized_roots):
            if is_in_scope(url, [root]):
                return i
        return None

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    robots_cache: dict[str, RobotFileParser] = {}
    # robots.txt crawl-delay can differ per host; honor the strictest one
    # among all roots rather than tracking a separate delay per host.
    effective_delay = delay
    for root_url in normalized_roots:
        rp = _get_robot_parser(root_url, robots_cache)
        effective_delay = max(effective_delay, rp.crawl_delay(user_agent) or 0)

    visited: set[str] = set(normalized_roots)
    queue: deque[str] = deque(normalized_roots)
    failures: list[str] = []
    skipped_robots: list[str] = []
    fetched_per_root = [0] * len(normalized_roots)

    with httpx.Client(follow_redirects=True, timeout=10.0, headers={"User-Agent": user_agent}) as client:
        while queue:
            url = queue.popleft()
            root_idx = root_index_for(url)
            if root_idx is not None and max_pages is not None and fetched_per_root[root_idx] >= max_pages:
                # This root's budget is exhausted — skip just this URL (no
                # fetch, so nothing further gets discovered through it), but
                # keep draining the queue for every other root.
                continue

            if not can_fetch(url, robots_cache, user_agent):
                skipped_robots.append(url)
                continue

            local_path = url_to_local_path(url, output_dir)
            if local_path.exists():
                html_text = local_path.read_text(encoding="utf-8")
                base_for_links = url
                if not url_sidecar_path(local_path).exists():
                    # Backfills the URL for a page saved by a crawl predating
                    # this sidecar, without re-fetching it over the network.
                    save_url_sidecar(local_path, url)
            else:
                time.sleep(effective_delay)
                result = fetch_and_save(url, output_dir, client, user_agent)
                if root_idx is not None:
                    fetched_per_root[root_idx] += 1
                if result is None:
                    failures.append(url)
                    continue
                saved_path, html_text, final_url = result
                if final_url != url:
                    if not is_in_scope(final_url, normalized_roots):
                        saved_path.unlink(missing_ok=True)
                        continue
                    visited.add(final_url)
                base_for_links = final_url

            for link in extract_links(html_text, base_for_links):
                if link not in visited and is_in_scope(link, normalized_roots):
                    visited.add(link)
                    queue.append(link)

    return {"failures": failures, "skipped_robots": skipped_robots}


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Uso: python -m ingestion.web_crawler <root_url>[,<root_url>...] [output_dir] [max_pages]")
        sys.exit(1)

    root_urls = [u.strip() for u in sys.argv[1].split(",") if u.strip()]
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else settings.data_raw_dir
    max_pages = int(sys.argv[3]) if len(sys.argv) > 3 else settings.crawl_max_pages

    result = crawl_site(root_urls, output_dir, max_pages=max_pages)
    if result["failures"]:
        print(f"Crawling completato con {len(result['failures'])} errore/i:")
        for failure in result["failures"]:
            print(f"  - {failure}")
    else:
        print(f"Crawling completato senza errori. Output in {output_dir}")
    if result["skipped_robots"]:
        print(f"{len(result['skipped_robots'])} URL saltati per robots.txt.")
