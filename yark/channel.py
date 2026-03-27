"""Channel and overall archive management with downloader"""

from __future__ import annotations
from datetime import datetime
import json
from pathlib import Path
import time
import threading
from yt_dlp import YoutubeDL, DownloadError  # type: ignore
from colorama import Style, Fore
import sys
from .reporter import Reporter
from .errors import ArchiveNotFoundException, _err_msg, VideoNotFoundException
from .video import Video, Element
from typing import Any
from progress.spinner import PieSpinner
from concurrent.futures import ThreadPoolExecutor, as_completed

MAX_METADATA_WORKERS = 64
"""Maximum number of concurrent metadata fetches"""

MAX_DOWNLOAD_WORKERS = 32
"""Maximum number of concurrent video downloads (each uses 128 fragment threads)"""

_shutdown_event = threading.Event()


def request_shutdown():
    """Signal all workers to stop gracefully"""
    _shutdown_event.set()


def is_shutting_down() -> bool:
    return _shutdown_event.is_set()

ARCHIVE_COMPAT = 3
"""
Version of Yark archives which this script is capable of properly parsing

- Version 1 was the initial format and had all the basic information you can see in the viewer now
- Version 2 introduced livestreams and shorts into the mix, as well as making the channel id into a simple url
- Version 3 was a minor change to introduce a deleted tag so we have full reporting capability

Some of these breaking versions are large changes and some are relatively small.
We don't check if a value exists or not in the archive format out of precedent
and we don't have optionally-present values, meaning that any new tags are a
breaking change to the format. The only downside to this is that the migrator
gets a line or two of extra code every breaking change. This is much better than
having way more complexity in the archiver decoding system itself.
"""

from typing import Optional


class DownloadConfig:
    max_videos: Optional[int]
    max_livestreams: Optional[int]
    max_shorts: Optional[int]
    skip_download: bool
    skip_metadata: bool
    format: Optional[str]
    max_age: Optional[float]

    def __init__(self) -> None:
        self.max_videos = None
        self.max_livestreams = None
        self.max_shorts = None
        self.skip_download = False
        self.skip_metadata = False
        self.format = None
        self.max_age = 24.0

    def submit(self):
        """Submits configuration, this has the effect of normalising maximums to 0 properly"""
        # Adjust remaining maximums if one is given
        no_maximums = (
            self.max_videos is None
            and self.max_livestreams is None
            and self.max_shorts is None
        )
        if not no_maximums:
            if self.max_videos is None:
                self.max_videos = 0
            if self.max_livestreams is None:
                self.max_livestreams = 0
            if self.max_shorts is None:
                self.max_shorts = 0

        # If all are 0 as its equivalent to skipping download
        if self.max_videos == 0 and self.max_livestreams == 0 and self.max_shorts == 0:
            print(
                Fore.YELLOW
                + "Using the skip downloads option is recommended over setting maximums to 0"
                + Fore.RESET
            )
            self.skip_download = True


class DownloadProgress:
    """Thread-safe tracker for concurrent download progress"""

    def __init__(self, total: int):
        self._lock = threading.Lock()
        self._total = total
        self._active = 0
        self._completed = 0

    def start(self):
        with self._lock:
            self._active += 1
            self._print()

    def finish(self):
        with self._lock:
            self._active -= 1
            self._completed += 1
            self._print()

    def _print(self):
        print(
            Style.DIM
            + f"  • Downloading: {self._active} active, {self._completed}/{self._total} completed"
            + "        "
            + Style.NORMAL,
            end="\r",
        )


class VideoLogger:
    def debug(self, msg):
        pass

    def info(self, msg):
        pass

    def warning(self, msg):
        pass

    def error(self, msg):
        pass


class Channel:
    path: Path
    version: int
    url: str
    videos: list[Video]
    livestreams: list[Video]
    shorts: list[Video]
    reporter: Reporter
    _video_index: dict[str, Video]
    _downloaded_cache: Optional[set[str]]

    @staticmethod
    def new(path: Path, url: str) -> Channel:
        """Creates a new channel"""
        print("Creating new channel..")
        channel = Channel()
        channel.path = Path(path)
        channel.version = ARCHIVE_COMPAT
        channel.url = url
        channel.videos = []
        channel.livestreams = []
        channel.shorts = []
        channel.reporter = Reporter(channel)
        channel._video_index = {}
        channel._downloaded_cache = None

        channel.commit()
        return channel

    @staticmethod
    def _new_empty() -> Channel:
        return Channel.new(
            Path("pretend"), "https://www.youtube.com/channel/UCSMdm6bUYIBN0KfS2CVuEPA"
        )

    @staticmethod
    def load(path: Path) -> Channel:
        """Loads existing channel from path"""
        # Check existence
        path = Path(path)
        channel_name = path.name
        print(f"Loading {channel_name} channel..")
        if not path.exists():
            raise ArchiveNotFoundException("Archive doesn't exist")

        with open(path / "yark.json", "r") as f:
            encoded = json.load(f)

        # Check version before fully decoding and exit if wrong
        archive_version = encoded["version"]
        if archive_version != ARCHIVE_COMPAT:
            encoded = _migrate_archive(
                archive_version, ARCHIVE_COMPAT, encoded, channel_name
            )

        # Decode and return
        return Channel._from_dict(encoded, path)

    def metadata(self, config: Optional[DownloadConfig] = None):
        """Queries YouTube for all channel metadata to refresh known videos"""
        cache_path = self.path / "metadata_cache.json"
        max_age = config.max_age if config is not None else 24.0

        if max_age is not None and cache_path.exists():
            cache_stat = cache_path.stat()
            age_hours = (time.time() - cache_stat.st_mtime) / 3600
            if age_hours < max_age:
                print(f"Using cached metadata ({age_hours:.1f}h old, max {max_age}h)..")
                with open(cache_path, "r") as f:
                    res = json.load(f)
                self._parse_metadata(res)
                self._save_descriptions()
                return

        print("Downloading metadata..")
        res = self._download_metadata()
        print()

        with open(cache_path, "w") as file:
            json.dump(res, file)

        self._parse_metadata(res)
        self._save_descriptions()

    def _download_metadata(self) -> dict[str, Any]:
        """Downloads metadata dict and returns for further parsing"""
        flat_settings = _metadata_settings(flat=True)
        video_settings = _metadata_settings(flat=False)

        res: Optional[dict[str, Any]] = None
        with YoutubeDL(flat_settings) as ydl:
            for i in range(3):
                try:
                    res = ydl.extract_info(self.url, download=False)
                    break
                except Exception as exception:
                    retrying = i != 2
                    _err_dl("metadata", exception, retrying)
                    if retrying:
                        print(
                            Style.DIM
                            + f"  • Retrying metadata download.."
                            + Style.RESET_ALL
                        )

        if res is None:
            _err_msg("Failed to download metadata after all retries", True)
            sys.exit(1)

        jobs: list[tuple] = []
        for index in range(len(res["entries"])):
            if res["entries"][index]["_type"] == "playlist":
                playlist = res["entries"][index]
                for list_index in range(len(playlist["entries"])):
                    url = playlist["entries"][list_index]["url"]
                    jobs.append(("playlist", index, list_index, url))
            elif res["entries"][index]["_type"] == "url":
                url = res["entries"][index]["url"]
                jobs.append(("url", index, None, url))

        progress = DownloadProgress(len(jobs))

        with ThreadPoolExecutor(max_workers=MAX_METADATA_WORKERS) as executor:
            futures = {
                executor.submit(_fetch_single_metadata, video_settings.copy(), url, progress): (kind, idx, sub_idx)
                for kind, idx, sub_idx, url in jobs
            }
            try:
                for future in as_completed(futures):
                    if _shutdown_event.is_set():
                        break
                    kind, idx, sub_idx = futures[future]
                    entry = future.result()
                    if entry is None:
                        continue
                    if kind == "playlist":
                        res["entries"][idx]["entries"][sub_idx] = entry
                    else:
                        res["entries"][idx] = entry
            except KeyboardInterrupt:
                _shutdown_event.set()
                for f in futures:
                    f.cancel()
                executor.shutdown(wait=False, cancel_futures=True)
                raise

        return res

    def _parse_metadata(self, res: dict[str, Any]):
        """Parses entirety of downloaded metadata"""
        # Normalize into types of videos
        videos = []
        livestreams = []
        shorts = []
        if len(res["entries"]) > 0 and "entries" not in res["entries"][0]:
            # Videos only
            videos = res["entries"]
        else:
            # Videos and at least one other (livestream/shorts)
            for entry in res["entries"]:
                kind = entry["title"].split(" - ")[-1].lower()
                if kind == "videos":
                    videos = entry["entries"]
                elif kind == "live":
                    livestreams = entry["entries"]
                elif kind == "shorts":
                    shorts = entry["entries"]
                else:
                    _err_msg(f"Unknown video kind '{kind}' found", True)

        # Parse metadata
        self._parse_metadata_videos("video", videos, self.videos)
        self._parse_metadata_videos("livestream", livestreams, self.livestreams)
        self._parse_metadata_videos("shorts", shorts, self.shorts)

        # Go through each and report deleted
        self._report_deleted(self.videos)
        self._report_deleted(self.livestreams)
        self._report_deleted(self.shorts)

    def download(self, config: DownloadConfig):
        """Downloads all videos which haven't already been downloaded"""
        self._build_download_cache()
        self._clean_parts()

        settings = {
            "outtmpl": f"{self.path}/videos/%(id)s.%(ext)s",
            "logger": VideoLogger(),
            "format": config.format if config.format is not None else "bestvideo+bestaudio/best",
            "merge_output_format": "mkv",
            "concurrent_fragment_downloads": 128,
            "retries": 10,
            "fragment_retries": 10,
            "socket_timeout": 60,
            "writesubtitles": True,
            "subtitleslangs": ["all"],
            "writeinfojson": True,
            "writethumbnail": False,
            "postprocessors": [{"key": "FFmpegEmbedSubtitle"}],
        }

        for i in range(5):
            try:
                not_downloaded = self._curate(config)

                if len(not_downloaded) == 0:
                    break

                if i == 0:
                    fmt_num = (
                        "a new video"
                        if len(not_downloaded) == 1
                        else f"{len(not_downloaded)} new videos"
                    )
                    print(f"Downloading {fmt_num}..")

                progress = DownloadProgress(len(not_downloaded))

                with ThreadPoolExecutor(max_workers=MAX_DOWNLOAD_WORKERS) as executor:
                    futures = {
                        executor.submit(_download_single_video, settings.copy(), video, progress): video
                        for video in not_downloaded
                    }
                    try:
                        for future in as_completed(futures):
                            if _shutdown_event.is_set():
                                break
                            status, video = future.result()
                            if status == "cancelled":
                                continue
                            elif status == "private":
                                print(
                                    "\n"
                                    + Style.DIM
                                    + f"  • Skipping {video.id} (deleted)"
                                    + Style.NORMAL,
                                    end="",
                                )
                                if video.deleted.current() == False:
                                    self.reporter.deleted.append(video)
                                    video.deleted.update(None, True)
                            elif status == "no_format":
                                print(
                                    "\n"
                                    + Fore.YELLOW
                                    + f"  • Skipping {video.id} (no downloadable format available)"
                                    + Fore.RESET,
                                    end="",
                                    file=sys.stderr,
                                )
                            elif status == "network_error":
                                print(
                                    "\n"
                                    + Fore.YELLOW
                                    + f"  • Skipping {video.id} (network error after retries)"
                                    + Fore.RESET,
                                    end="",
                                    file=sys.stderr,
                                )
                    except KeyboardInterrupt:
                        _shutdown_event.set()
                        for f in futures:
                            f.cancel()
                        executor.shutdown(wait=False, cancel_futures=True)
                        raise

                print()

                break

            except Exception as exception:
                if i == 0:
                    print()
                _err_dl("videos", exception, i != 4)

    def search(self, id: str):
        """Searches channel for a video with the corresponding `id` and returns"""
        video = self._video_index.get(id)
        if video is not None:
            return video
        raise VideoNotFoundException(f"Couldn't find {id} inside archive")

    def _curate(self, config: DownloadConfig) -> list[Video]:
        """Curate videos which aren't downloaded and return their urls"""

        def curate_list(videos: list[Video], maximum: Optional[int]) -> list[Video]:
            """Curates the videos inside of the provided `videos` list to it's local maximum"""
            # Cut available videos to maximum if present for deterministic getting
            if maximum is not None:
                # Fix the maximum to the length so we don't try to get more than there is
                fixed_maximum = min(max(len(videos) - 1, 0), maximum)

                # Set the available videos to this fixed maximum
                new_videos = []
                for ind in range(fixed_maximum):
                    new_videos.append(videos[ind])
                videos = new_videos

            # Find undownloaded videos in available list
            not_downloaded = []
            for video in videos:
                if not video.downloaded():
                    not_downloaded.append(video)

            # Return
            return not_downloaded

        # Curate
        not_downloaded = []
        not_downloaded.extend(curate_list(self.videos, config.max_videos))
        not_downloaded.extend(curate_list(self.livestreams, config.max_livestreams))
        not_downloaded.extend(curate_list(self.shorts, config.max_shorts))

        # Return
        return not_downloaded

    def commit(self):
        """Commits (saves) archive to path; do this once you've finished all of your transactions"""
        # Save backup
        self._backup()

        # Directories
        print(f"Committing {self} to file..")
        paths = [self.path, self.path / "thumbnails", self.path / "videos"]
        for path in paths:
            if not path.exists():
                path.mkdir()

        # Config
        with open(self.path / "yark.json", "w+") as file:
            json.dump(self._to_dict(), file)

    def _parse_metadata_videos(self, kind: str, i: list, bucket: list):
        """Parses metadata for a category of video into it's bucket and tells user what's happening"""

        # Print at the start without loading indicator so theres always a print
        msg = f"Parsing {kind} metadata.."
        print(msg, end="\r")

        # Start computing and show loading spinner
        with ThreadPoolExecutor() as ex:
            # Make future for computation of the video list
            future = ex.submit(self._parse_metadata_videos_comp, i, bucket)

            # Start spinning
            with PieSpinner(f"{msg} ") as bar:
                # Don't show bar for 2 seconds but check if future is done
                no_bar_time = time.time() + 2
                while time.time() < no_bar_time:
                    if future.done():
                        return
                    time.sleep(0.25)

                # Spin until future is done
                while not future.done():
                    time.sleep(0.075)
                    bar.next()

    def _parse_metadata_videos_comp(self, i: list, bucket: list):
        """Computes the actual parsing for `_parse_metadata_videos` without outputting what's happening"""
        bucket_index = {v.id: v for v in bucket}
        for entry in i:
            if "formats" not in entry or len(entry["formats"]) == 0:
                continue

            existing = bucket_index.get(entry["id"])
            if existing is not None:
                existing.update(entry)
            else:
                video = Video.new(entry, self)
                bucket.append(video)
                bucket_index[video.id] = video
                self._video_index[video.id] = video
                self.reporter.added.append(video)

        bucket.sort(reverse=True)

    def _save_descriptions(self):
        """Saves current description for each video as a text file"""
        desc_dir = self.path / "descriptions"
        if not desc_dir.exists():
            desc_dir.mkdir()
        for bucket in [self.videos, self.livestreams, self.shorts]:
            for video in bucket:
                desc = video.description.current()
                if desc is None:
                    continue
                desc_path = desc_dir / f"{video.id}.txt"
                if desc_path.exists():
                    try:
                        existing = desc_path.read_text(encoding="utf-8")
                        if existing == desc:
                            continue
                    except Exception:
                        pass
                with open(desc_path, "w", encoding="utf-8") as f:
                    f.write(desc)

    def _report_deleted(self, videos: list):
        """Goes through a video category to report & save those which where not marked in the metadata as deleted if they're not already known to be deleted"""
        for video in videos:
            if video.deleted.current() == False and not video.known_not_deleted:
                self.reporter.deleted.append(video)
                video.deleted.update(None, True)

    def _build_download_cache(self):
        """Builds a set of downloaded video stems for O(1) lookups"""
        videos_dir = self.path / "videos"
        if not videos_dir.exists():
            self._downloaded_cache = set()
            return
        self._downloaded_cache = {
            f.stem for f in videos_dir.iterdir()
            if f.suffix in Video._VIDEO_EXTENSIONS
        }

    def _clean_parts(self):
        """Cleans old temporary `.part` files which where stopped during download if present"""
        videos = self.path / "videos"
        if not videos.exists():
            return

        deletion_bucket: list[Path] = []
        for file in videos.iterdir():
            if file.suffix == ".part" or file.suffix == ".ytdl":
                deletion_bucket.append(file)

        if len(deletion_bucket) != 0:
            print("Cleaning out previous temporary files..")
            for file in deletion_bucket:
                file.unlink()

    def _backup(self):
        """Creates a backup of the existing `yark.json` file in path as `yark.bak` with added comments"""
        # Get current archive path
        ARCHIVE_PATH = self.path / "yark.json"

        # Skip backing up if the archive doesn't exist
        if not ARCHIVE_PATH.exists():
            return

        # Open original archive to copy
        with open(self.path / "yark.json", "r") as file_archive:
            # Add comment information to backup file
            save = f"// Backup of a Yark archive, dated {datetime.utcnow().isoformat()}\n// Remove these comments and rename to 'yark.json' to restore\n{file_archive.read()}"

            # Save new information into a new backup
            with open(self.path / "yark.bak", "w+") as file_backup:
                file_backup.write(save)

    @staticmethod
    def _from_dict(encoded: dict, path: Path) -> Channel:
        """Decodes archive which is being loaded back up"""
        channel = Channel()
        channel.path = path
        channel.version = encoded["version"]
        channel.url = encoded["url"]
        channel.reporter = Reporter(channel)
        channel._downloaded_cache = None
        channel.videos = [
            Video._from_dict(video, channel) for video in encoded["videos"]
        ]
        channel.livestreams = [
            Video._from_dict(video, channel) for video in encoded["livestreams"]
        ]
        channel.shorts = [
            Video._from_dict(video, channel) for video in encoded["shorts"]
        ]
        channel._video_index = {}
        for v in channel.videos:
            channel._video_index[v.id] = v
        for v in channel.livestreams:
            channel._video_index[v.id] = v
        for v in channel.shorts:
            channel._video_index[v.id] = v
        return channel

    def _to_dict(self) -> dict:
        """Converts channel data to a dictionary to commit"""
        return {
            "version": self.version,
            "url": self.url,
            "videos": [video._to_dict() for video in self.videos],
            "livestreams": [video._to_dict() for video in self.livestreams],
            "shorts": [video._to_dict() for video in self.shorts],
        }

    def __repr__(self) -> str:
        return self.path.name


def _metadata_settings(flat: bool = False) -> dict:
    """Returns yt-dlp settings for metadata downloading"""
    settings = {
        "logger": VideoLogger(),
        "ignore_no_formats_error": True,
    }
    if flat:
        settings["extract_flat"] = True
    return settings


def _fetch_single_metadata(settings: dict, url: str, progress: Optional[DownloadProgress] = None) -> Any:
    """Fetches metadata for a single video URL with retries; each call uses its own YoutubeDL instance for thread safety"""
    if _shutdown_event.is_set():
        return None
    if progress:
        progress.start()
    try:
        for i in range(3):
            if _shutdown_event.is_set():
                return None
            try:
                with YoutubeDL(settings) as ydl:
                    entry = ydl.extract_info(url, download=False)
                    if entry.get("formats") is not None and len(entry["formats"]) == 0:
                        with YoutubeDL(settings) as ydl2:
                            entry = ydl2.extract_info(url, download=False)
                    return entry
            except Exception as exception:
                if _shutdown_event.is_set():
                    return None
                retrying = i != 2
                _err_dl("metadata", exception, retrying)
                if retrying:
                    print(
                        Style.DIM
                        + f"  • Retrying metadata download.."
                        + Style.RESET_ALL
                    )
        return None
    finally:
        if progress:
            progress.finish()


def _download_single_video(settings: dict, video, progress: DownloadProgress) -> tuple[str, Any]:
    """Downloads a single video file; returns (status, video) where status is 'ok', 'private', 'no_format', 'cancelled', or 'error'"""
    if _shutdown_event.is_set():
        return ("cancelled", video)
    progress.start()
    try:
        for attempt in range(3):
            if _shutdown_event.is_set():
                return ("cancelled", video)
            try:
                with YoutubeDL(settings) as ydl:
                    ydl.download([video.url()])
                return ("ok", video)
            except DownloadError as exception:
                if _shutdown_event.is_set():
                    return ("cancelled", video)
                if (
                    "Private video" in exception.msg
                    or "This video has been removed by the uploader" in exception.msg
                ):
                    return ("private", video)
                elif "Requested format is not available" in exception.msg:
                    return ("no_format", video)
                elif (
                    "timed out" in exception.msg
                    or "bytes read" in exception.msg
                    or "Connection reset" in exception.msg
                    or "HTTPSConnectionPool" in exception.msg
                    or "Read timed out" in exception.msg
                    or "incomplete read" in exception.msg.lower()
                ):
                    if attempt < 2:
                        time.sleep(5 * (attempt + 1))
                        continue
                    return ("network_error", video)
                else:
                    raise
        return ("no_format", video)
    finally:
        progress.finish()


def _skip_video(
    videos: list[Video],
    reason: str,
    warning: bool = False,
) -> tuple[list[Video], Video]:
    """Skips first undownloaded video in `videos`, make sure there's at least one to skip otherwise an exception will be thrown"""
    # Find fist undownloaded video
    for ind, video in enumerate(videos):
        if not video.downloaded():
            # Tell the user we're skipping over it
            if warning:
                print(
                    Fore.YELLOW + f"  • Skipping {video.id} ({reason})" + Fore.RESET,
                    file=sys.stderr,
                )
            else:
                print(
                    Style.DIM + f"  • Skipping {video.id} ({reason})" + Style.NORMAL,
                )

            # Set videos to skip over this one
            videos = videos[ind + 1 :]

            # Return the corrected list and the video found
            return videos, video

    # Shouldn't happen, see docs
    raise Exception(
        "We expected to skip a video and return it but nothing to skip was found"
    )


def _migrate_archive(
    current_version: int, expected_version: int, encoded: dict, channel_name: str
) -> dict:
    """Automatically migrates an archive from one version to another by bootstrapping"""

    def migrate_step(cur: int, encoded: dict) -> dict:
        """Step in recursion to migrate from one to another, contains migration logic"""
        # Stop because we've reached the desired version
        if cur == expected_version:
            return encoded

        # From version 1 to version 2
        elif cur == 1:
            # Channel id to url
            encoded["url"] = "https://www.youtube.com/channel/" + encoded["id"]
            del encoded["id"]
            print(
                Fore.YELLOW
                + "Please make sure "
                + encoded["url"]
                + " is the correct url"
                + Fore.RESET
            )

            # Empty livestreams/shorts lists
            encoded["livestreams"] = []
            encoded["shorts"] = []

        # From version 2 to version 3
        elif cur == 2:
            # Add deleted status to every video/livestream/short
            # NOTE: none is fine for new elements, just a slight bodge
            for video in encoded["videos"]:
                video["deleted"] = Element.new(Video._new_empty(), False)._to_dict()
            for video in encoded["livestreams"]:
                video["deleted"] = Element.new(Video._new_empty(), False)._to_dict()
            for video in encoded["shorts"]:
                video["deleted"] = Element.new(Video._new_empty(), False)._to_dict()

        # Unknown version
        else:
            _err_msg(f"Unknown archive version v{cur} found during migration", True)
            sys.exit(1)

        # Increment version and run again until version has been reached
        cur += 1
        encoded["version"] = cur
        return migrate_step(cur, encoded)

    # Inform user of the backup process
    print(
        Fore.YELLOW
        + f"Automatically migrating archive from v{current_version} to v{expected_version}, a backup has been made at {channel_name}/yark.bak"
        + Fore.RESET
    )

    # Start recursion step
    return migrate_step(current_version, encoded)


def _err_dl(name: str, exception: DownloadError, retrying: bool):
    """Prints errors to stdout depending on what kind of download error occurred"""
    # Default message
    msg = f"Unknown error whilst downloading {name}, details below:\n{exception}"

    # Types of errors
    ERRORS = [
        "<urlopen error [Errno 8] nodename nor servname provided, or not known>",
        "500",
        "Got error: The read operation timed out",
        "No such file or directory",
        "HTTP Error 404: Not Found",
        "<urlopen error timed out>",
    ]

    # Download errors
    if type(exception) == DownloadError:
        # Server connection
        if ERRORS[0] in exception.msg:
            msg = "Issue connecting with YouTube's servers"

        # Server fault
        elif ERRORS[1] in exception.msg:
            msg = "Fault with YouTube's servers"

        # Timeout
        elif ERRORS[2] in exception.msg:
            msg = "Timed out trying to download video"

        # Video deleted whilst downloading
        elif ERRORS[3] in exception.msg:
            msg = "Video deleted whilst downloading"

        # Channel not found, might need to retry with alternative route
        elif ERRORS[4] in exception.msg:
            msg = "Couldn't find channel by it's id"

        # Random timeout; not sure if its user-end or youtube-end
        elif ERRORS[5] in exception.msg:
            msg = "Timed out trying to reach YouTube"

    # Print error
    suffix = ", retrying in a few seconds.." if retrying else ""
    print(
        Fore.YELLOW + "  • " + msg + suffix.ljust(40) + Fore.RESET,
        file=sys.stderr,
    )

    # Wait if retrying, exit if failed
    if retrying:
        time.sleep(5)
    else:
        _err_msg(f"  • Sorry, failed to download {name}", True)
        sys.exit(1)
