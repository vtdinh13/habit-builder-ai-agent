import requests
import xml.etree.ElementTree as ET
import io

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from tqdm import tqdm
import os
import logging
import pandas as pd

import psycopg
from psycopg import Connection, sql
from psycopg.rows import dict_row

from faster_whisper import WhisperModel

model = WhisperModel(model_size_or_path='tiny', device='cpu', compute_type='int8')

logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s")

def fetch_rss_feed(rss_url:str, outpath:Path):
    meta = []
    categories = []
    namespaces = {}

    xml = requests.get(rss_url).text

    for event, elem in ET.iterparse(io.StringIO(xml), events=('start-ns',)):
        prefix, uri = elem
        namespaces[prefix] = uri
    
    # itunes_namespace = namespaces


    root = ET.fromstring(xml)
    name_of_podcast = root.findtext('channel/title')
    language = root.findtext('channel/language')

    
    for cat in root.findall('channel/itunes:category', namespaces=namespaces):
        main_cat = cat.get('text')
        subcats = [sub.get('text') for sub in cat.findall('itunes:category', namespaces=namespaces)]
        categories.append({
            'topic': main_cat,
            'subtopic': subcats
        })

    for i, item in enumerate(root.findall('channel/item')):
        ep_name = item.findtext('title')
        media_url = item.find('enclosure')
        duration = item.findtext('itunes:duration', namespaces=namespaces)
        pubDate = item.findtext('pubDate')
        
        meta.append(
            {'name_of_podcast': name_of_podcast, 
            'categories': categories,
            'language': language,
            'ep_name': ep_name, 
            'pub_date': pubDate,
            'duration': duration,
            'media_url': media_url.get('url')}
    )
    
    with open(outpath, 'w', encoding='utf-8') as f_out:
        json.dump(meta, f_out)
        
    return 'SUCCESS'



def make_queue(rss_path: Path, media_directory: Path) -> List[Dict[str, Any]]:
    """
    Build a JSON-serializable list of RSS episodes that still need downloads/transcriptions.

    Args:
        rss_path: Path object pointing to the JSON file produced by `fetch_rss_feed`.
        media_directory: Directory that stores downloaded media and transcripts. Existing filenames
            are compared against RSS entries to determine missing episodes.

    Returns:
        List[Dict[str, Any]]: one dict per missing episode containing all metadata fields required
        by downstream download/transcription steps.
    """

    rss_path = Path(rss_path)
    media_directory = Path(media_directory)
    media_directory.mkdir(exist_ok=True)

    directory_list = os.listdir(media_directory)
    directory_list_names = [Path(i).stem for i in directory_list]

    rss_df = pd.read_json(rss_path)
    rss_list = [i for i in rss_df['ep_name']]

    to_download = set(rss_list) - set(directory_list_names)
    to_download_list = list(to_download)

    rss_to_download = rss_df[rss_df["ep_name"].isin(to_download_list)]

    # Convert the filtered dataframe to plain Python dicts so the queue is JSON serializable.
    download_queue = rss_to_download.to_dict(orient='records')

    print(f"Total number of files left to process: {len(download_queue)}")
    return download_queue



def sanitize_episode_name(name: str) -> str:
    """
    Convert an episode name into a filesystem-safe stem.
    """
    safe = re.sub(r'[\\/:*?"<>|]+', '_', name)
    return safe.strip().strip(".")


def download_media_file(media_file: dict, media_directory: Path) -> None:
    """
    Stream the media file defined in the RSS entry into the local media directory.

    Args:
        media_file: Dictionary describing the episode (expects `ep_name` and `media_url`).
        media_directory: Directory where the mp3 file should be saved.

    Returns:
        None. Writes the file to disk and logs success/failure.
    """

    media_directory = Path(media_directory)
    media_directory.mkdir(parents=True, exist_ok=True)

    safe_name = sanitize_episode_name(media_file['ep_name'])
    media_url = media_file['media_url']
    audiofile = media_directory / f"{safe_name}.mp3"


    with requests.get(media_url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))

        with audiofile.open('wb') as f_out, tqdm(total=total, unit='B', unit_scale=True, desc=str(audiofile)) as bar:
            for chunk in r.iter_content(1024 * 1024):
                f_out.write(chunk)
                bar.update(len(chunk))

    if audiofile.exists() and audiofile.stat().st_size > 0:
        print(f"Download successful: {audiofile}.")
    else:
        print(f"Download not successful: {audiofile}")

def format_time(seconds: float) -> str:
    """
    Convert a duration in seconds into HH:MM:SS format for transcript timestamps.

    Args:
        seconds: Floating-point duration returned by the transcription model.

    Returns:
        Timestamp string padded to hours:minutes:seconds.
    """
    sec = int(seconds)
    hour, remainder = divmod(sec, 3600)
    min, sec = divmod(remainder, 60)
    return f"{hour:02d}:{min:02d}:{sec:02d}"


def transcribe_audio_file(media_directory: Path) -> str:
    """
    Generate text transcripts for every mp3 file in the target directory using Whisper.

    Args:
        media_directory: Folder containing the mp3 files to process.
        model_size_or_path: Faster-Whisper model variant or local path.
        device: Compute device passed to Faster-Whisper (`cpu` or `cuda`).
        compute_type: Precision used during inference (e.g. `int8`, `float16`).

    Returns:
        The literal string 'SUCCESS' after all files are transcribed.
    """
    media_directory = Path(media_directory)
    media_directory.mkdir(parents=True, exist_ok=True)
    audiofile = sorted(media_directory.glob("*.mp3"))

    if not audiofile:
        print(f"No audio files found in {media_directory}.")

    for f in audiofile:
        segments, info = model.transcribe(str(f), vad_filter=True)
        name_of_episode = f.stem

        transcript_path = media_directory / f"{name_of_episode}.txt"

        with transcript_path.open('w', encoding='utf-8') as f_out, \
        tqdm(total=float(info.duration), unit='s', desc=f"Transcribing: {name_of_episode}") as bar:
            for s in segments:
                line = f"({format_time(s.start)}) {s.text.strip()}"
                f_out.write(line + '\n')
                f_out.flush()  # ensures that data is physically written to disk and not held in memory

                bar.n = min(s.end, info.duration)
                bar.refresh()
            bar.n = bar.total
            bar.refresh()

    return 'SUCCESS'


def fetch_episode_metadata(conn: Connection, episode_name: str) -> Dict[str, str]:
    """
    Fetch the RSS metadata needed for storing transcripts.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT id, ep_name, name_of_podcast
            FROM rss
            WHERE ep_name = %s;
            """,
            (episode_name,),
        )
        row = cur.fetchone()

    if row is None:
        raise ValueError(f"No RSS row found with name: {episode_name}")
    return row

def insert_transcript(
    conn: Connection,
    table_name: str,
    rss_id: int,
    name_of_podcast: str,
    ep_name: str,
    transcript: str,
):
    """
    Insert the transcript into the specified table.
    """
    with conn.cursor() as cur:
        query = sql.SQL(
            """
            INSERT INTO {table} (
                rss_id, name_of_podcast, ep_name, transcript
            )
            SELECT %s, %s, %s, %s
            WHERE NOT EXISTS (
                SELECT 1 FROM {table} WHERE rss_id = %s
            );
            """
        ).format(table=sql.Identifier(table_name))
        cur.execute(
            query,
            (rss_id, name_of_podcast, ep_name, transcript, rss_id)
        )
        if cur.rowcount:
            print('Inserted new transcript for episode:', ep_name)
        else:
            print('Duplicate entry detected; skipped insert.')

def ingest(
    rss_path: Path,
    media_directory: Path,
    transcript_table: str, 
    database_url: str,
    limit: Optional[int] = None
) -> str:
    """
    Download any pending RSS episodes and generate transcripts for the specified directory.

    Args:
        rss_path: Path object pointing to the RSS JSON used to determine pending episodes.
        media_directory: Directory where audio files and transcripts are stored.
        limit: Optional max number of new episodes to process.
        model_size_or_path: Faster-Whisper model variant or path.
        device: Compute device passed to Faster-Whisper (`cpu` or `cuda`).
        compute_type: Precision used during inference (e.g. `int8`, `float16`).

    Returns:
        The literal string 'SUCCESS' once downloads/transcriptions finish.
    """

    media_directory = Path(media_directory)
    media_directory.mkdir(parents=True, exist_ok=True)

    download_queue = make_queue(rss_path, media_directory)

    if limit is not None:
        download_queue = download_queue[:limit]
        print(f"Remaining number of files to process in this run: {len(download_queue)}")

    if not download_queue:
        print("No new episodes to ingest.")
     
    downloaded_transcribed_count = 0
    postgres_count = 0
    skipped = []

    for episode in download_queue:
        safe_name = sanitize_episode_name(episode['ep_name'])
        audio_file_path = media_directory / f"{safe_name}.mp3"
        transcript_path = media_directory / f"{safe_name}.txt"
        try:
            download_media_file(media_file=episode, media_directory=media_directory)

            transcribe_audio_file(media_directory=media_directory)

            if transcript_path.exists() and transcript_path.stat().st_size > 0 and audio_file_path.exists():
                audio_file_path.unlink()
                downloaded_transcribed_count +=1
                print(f"Number of files processed in current run: {downloaded_transcribed_count}/{len(download_queue)}")
                print(f'Deleted: {audio_file_path}')
            
            with psycopg.connect(database_url, autocommit=False) as conn:
                print(f"Preparing to write transcript to postgres: {transcript_path}")
                metadata = fetch_episode_metadata(conn, episode["ep_name"])
                
                with transcript_path.open("r", encoding="utf-8") as f_in:
                        transcript_text = f_in.read()
                insert_transcript(
                            conn=conn,
                            table_name=transcript_table,
                            rss_id=metadata["id"],
                            name_of_podcast=metadata["name_of_podcast"],
                            ep_name=metadata["ep_name"],
                            transcript=transcript_text,
                        )
                conn.commit()
                postgres_count +=1
                print(f"Success writing transcript to postgres: {metadata['name_of_podcast']} : {metadata['ep_name']}")
        except Exception as exc:  # pylint: disable=broad-except
            logging.exception("Skipping episode due to error: %s (safe name: %s)", episode.get("ep_name"), safe_name)
            skipped.append({"ep_name": episode.get("ep_name", ""), "safe_name": safe_name, "error": str(exc)})
            continue
    print(f"Total number of files transcribed: {downloaded_transcribed_count}")
    print(f"Total number of transscripts written to postgres: {postgres_count}")
    if skipped:
        logging.warning("Skipped %d episodes due to errors.", len(skipped))
        for item in skipped:
            logging.warning("Skipped: %s (safe name: %s) error: %s", item["ep_name"], item["safe_name"], item["error"])
    return "SUCCESS"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Download new podcast episodes and generate transcripts.',
    )
    parser.add_argument(
        '--rss-path',
        type=Path,
        required=True,
        help='Path to the RSS JSON file containing episode metadata.',
    )
    parser.add_argument(
        '--media-dir',
        type=Path,
        required=True,
        help='Directory where media files and transcripts are stored.',
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Maximum number of episodes to process (default: no limit).',
    )
    parser.add_argument(
        '--transcript-table',
        type=str,
        default="transcripts",
        help='Postgres table name where transcripts will be stored.',
    )
    parser.add_argument(
        '--database-url',
        type=str,
        default="postgresql://podcast:podcast@localhost:5434/podcast-agent",
        help='Postgres connection string for storing transcripts.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ingest(
        rss_path=args.rss_path,
        media_directory=args.media_dir,
        transcript_table=args.transcript_table,
        database_url=args.database_url,
        limit=args.limit
    )


if __name__ == '__main__':
    main()
