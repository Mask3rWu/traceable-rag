"""Load canonical chunk data only when text or provenance is needed."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from src.retrieval.contracts import SearchResult
from src.retrieval.indexing import SourceChunk, discover_chunk_files, load_chunks


class ChunkCatalog:
    def __init__(self, sources: Sequence[SourceChunk]) -> None:
        self._sources = {source.chunk.chunk_id: source for source in sources}
        if len(self._sources) != len(sources):
            raise ValueError("Chunk catalog contains duplicate chunk IDs")

    @classmethod
    def load(cls, files: Sequence[Path] | None = None) -> "ChunkCatalog":
        return cls(load_chunks(discover_chunk_files(files)))

    def source(self, chunk_id: str) -> SourceChunk:
        try:
            return self._sources[chunk_id]
        except KeyError as exc:
            raise KeyError(f"Unknown chunk ID: {chunk_id}") from exc

    def block_ids(self, chunk_id: str) -> set[str]:
        return set(self.source(chunk_id).chunk.block_ids)

    def validate_result(self, result: SearchResult) -> None:
        source = self.source(result.chunk_id)
        if source.content_hash != result.content_hash:
            raise RuntimeError(
                f"Stale retrieval index for chunk {result.chunk_id}: "
                f"catalog={source.content_hash}, result={result.content_hash}"
            )

    @property
    def block_id_universe(self) -> set[str]:
        return {
            block_id
            for source in self._sources.values()
            for block_id in source.chunk.block_ids
        }

    def __len__(self) -> int:
        return len(self._sources)
