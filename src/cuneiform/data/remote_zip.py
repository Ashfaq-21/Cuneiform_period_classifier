"""Random access to members of a large remote ZIP archive over HTTP Range requests.

HeiCuBeDa ships its 3D scans as twenty 5-35 GB zip archives. The server supports
``Range`` requests, so we can read the zip central directory (at the end of the
file) and then stream out a single member without downloading the whole archive.

Only the standard library is used so this works before heavy deps are installed.
"""

from __future__ import annotations

import io
import shutil
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

USER_AGENT = "cuneiform-period-classifier/0.1 (research prototype)"


class HttpRangeFile(io.RawIOBase):
    """A read-only, seekable file object backed by HTTP Range requests."""

    def __init__(self, url: str, timeout: float = 120.0, retries: int = 5) -> None:
        super().__init__()
        self.url = url
        self.timeout = timeout
        self.retries = retries
        self._pos = 0
        # Some Dataverse front-ends reject HEAD, so probe with a 1-byte GET.
        with self._open_range(0, 0) as response:
            content_range = response.headers.get("Content-Range")
            if response.status != 206 or not content_range:
                raise OSError(
                    f"Server for {url} did not honour a Range request (HTTP {response.status}). "
                    "Download the archive manually and use local .ply files instead."
                )
            self.size = int(content_range.rsplit("/", 1)[-1])
            self.resolved_url = response.url  # follow redirects once
            response.read()

    def _open_range(self, start: int, end: int):
        url = getattr(self, "resolved_url", self.url)
        request = urllib.request.Request(
            url, headers={"Range": f"bytes={start}-{end}", "User-Agent": USER_AGENT}
        )
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                return urllib.request.urlopen(request, timeout=self.timeout)
            except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
                last_error = error
                time.sleep(2**attempt)
        raise OSError(f"Range request {start}-{end} to {url} failed: {last_error}")

    # io.RawIOBase interface ------------------------------------------------
    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        else:
            self._pos = self.size + offset
        return self._pos

    def readinto(self, buffer) -> int:  # type: ignore[override]
        if self._pos >= self.size or len(buffer) == 0:
            return 0
        end = min(self._pos + len(buffer), self.size) - 1
        with self._open_range(self._pos, end) as response:
            data = response.read()
        buffer[: len(data)] = data
        self._pos += len(data)
        return len(data)


def open_remote_zip(url: str, buffer_size: int = 8 << 20) -> zipfile.ZipFile:
    """Open a remote zip archive for listing and member extraction."""
    raw = HttpRangeFile(url)
    return zipfile.ZipFile(io.BufferedReader(raw, buffer_size=buffer_size))


def list_remote_zip(url: str) -> list[dict]:
    """Return ``[{name, compressed_size, size}]`` for every member of a remote zip."""
    with open_remote_zip(url, buffer_size=1 << 20) as archive:
        return [
            {"name": info.filename, "compressed_size": info.compress_size, "size": info.file_size}
            for info in archive.infolist()
            if not info.is_dir()
        ]


def extract_remote_member(url: str, member: str, destination: Path) -> Path:
    """Stream one member of a remote zip to ``destination`` atomically (via a .part file)."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    with open_remote_zip(url) as archive, archive.open(member) as source, open(partial, "wb") as sink:
        shutil.copyfileobj(source, sink, length=8 << 20)
    _replace_with_retry(partial, destination)
    return destination


def _replace_with_retry(source: Path, destination: Path, attempts: int = 10) -> None:
    """Rename, retrying while another process (often an antivirus scan on Windows) holds the file."""
    for attempt in range(attempts):
        try:
            source.replace(destination)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(min(2.0 * (attempt + 1), 10.0))
