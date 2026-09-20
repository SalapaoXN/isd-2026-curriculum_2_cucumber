"""Generate the unlabeled local semantic-threshold development review set."""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from rag.retrieval.embedder import EMBEDDING_DIMENSION, MODEL_NAME, embed_texts


TOPICS = (
    "computer networks and network protocols",
    "database systems and SQL",
    "artificial intelligence and machine learning",
    "web application development",
    "information security and cybersecurity",
    "programming and software development",
    "data structures and algorithms",
    "statistics and probability",
)


def _distance(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    query_norm = np.linalg.norm(query)
    vector_norms = np.linalg.norm(vectors, axis=1)
    if not np.isfinite(query_norm) or query_norm == 0:
        raise ValueError("topic embedding is non-finite or zero-norm")
    if not np.all(np.isfinite(vectors)) or np.any(vector_norms == 0):
        raise ValueError("description embedding is non-finite or zero-norm")
    return 1.0 - (vectors @ query) / (vector_norms * query_norm)


def _load_descriptions(db_path: Path) -> list[dict]:
    connection = sqlite3.connect(db_path)
    try:
        course_names = {
            row[0]: (row[2] or row[1] or "")
            for row in connection.execute(
                "SELECT course_id, name_th, name_en FROM courses ORDER BY course_id"
            )
        }
        records = []
        seen_courses: set[int] = set()
        for (payload,) in connection.execute(
            "SELECT chunk_json FROM semantic_chunks ORDER BY chunk_id"
        ):
            chunk = json.loads(payload)
            if chunk.get("chunk_type") != "description":
                continue
            course_id = chunk.get("course_id")
            if not isinstance(course_id, int) or course_id in seen_courses:
                continue
            text = chunk.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            seen_courses.add(course_id)
            records.append(
                {
                    "course_id": course_id,
                    "course_code": chunk.get("course_code"),
                    "course_name": course_names.get(course_id, ""),
                    "chunk_id": chunk["chunk_id"],
                    "description": text,
                }
            )
        return records
    finally:
        connection.close()


def generate(db_path: Path, output_path: Path) -> None:
    descriptions = _load_descriptions(db_path)
    if not descriptions:
        raise ValueError("runtime DB contains no usable description chunks")
    vectors = np.asarray(embed_texts([row["description"] for row in descriptions]), dtype=np.float64)
    if vectors.shape != (len(descriptions), EMBEDDING_DIMENSION):
        raise ValueError("description embeddings do not have the frozen dimension")

    topic_vectors = np.asarray(embed_texts(list(TOPICS)), dtype=np.float64)
    pairs = []
    for topic_index, (topic, query_vector) in enumerate(zip(TOPICS, topic_vectors)):
        distances = _distance(query_vector, vectors)
        order = sorted(range(len(descriptions)), key=lambda i: (float(distances[i]), descriptions[i]["chunk_id"]))
        selected = (order[0], order[len(order) // 2], order[-1])
        for pair_index, description_index in enumerate(selected, start=1):
            distance = float(distances[description_index])
            if not math.isfinite(distance):
                raise ValueError("computed distance is non-finite")
            row = descriptions[description_index]
            pairs.append(
                {
                    "id": f"stdev_{topic_index + 1:02d}_{pair_index:02d}",
                    "topic": topic,
                    "course_id": row["course_id"],
                    "course_code": row["course_code"],
                    "course_name": row["course_name"],
                    "chunk_id": row["chunk_id"],
                    "description": row["description"],
                    "distance": distance,
                    "relevant": None,
                    "embedding_model": MODEL_NAME,
                    "embedding_dimension": EMBEDDING_DIMENSION,
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "version": "semantic_threshold_dev_v1",
                "embedding_model": MODEL_NAME,
                "embedding_dimension": EMBEDDING_DIMENSION,
                "pairs": pairs,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=Path("cucumber_outputs/runtime/curriculum.db"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/fixtures/semantic_threshold_dev_v1.json"),
    )
    args = parser.parse_args()
    generate(args.db, args.output)


if __name__ == "__main__":
    main()
