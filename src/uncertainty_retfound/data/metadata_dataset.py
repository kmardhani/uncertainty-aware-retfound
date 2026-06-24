"""Generic metadata-backed image dataset utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from PIL import Image
import pandas as pd
from torch.utils.data import Dataset


def load_prepared_metadata(metadata: pd.DataFrame | str | Path) -> pd.DataFrame:
    """Load prepared metadata from a dataframe or CSV path."""

    if isinstance(metadata, pd.DataFrame):
        return metadata.copy()

    metadata_path = Path(metadata)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Prepared metadata CSV not found: {metadata_path}")

    return pd.read_csv(metadata_path)


class MetadataImageDataset(Dataset[dict[str, Any]]):
    """Generic image dataset that uses metadata `image_path` values as stored."""

    def __init__(
        self,
        metadata: pd.DataFrame | str | Path,
        image_root: str | Path,
        *,
        id_column: str = "id_code",
        image_path_column: str = "image_path",
        label_column: str = "label",
        transform: Callable[[Image.Image], Any] | None = None,
        validate_paths: bool = True,
    ) -> None:
        self.id_column = id_column
        self.image_path_column = image_path_column
        self.label_column = label_column
        self.image_root = Path(image_root)
        self.transform = transform

        loaded_metadata = load_prepared_metadata(metadata)

        for required_column in (self.id_column, self.image_path_column, self.label_column):
            if required_column not in loaded_metadata.columns:
                raise KeyError(
                    f"Metadata is missing required column: {required_column}. "
                    f"Available columns: {list(loaded_metadata.columns)}"
                )

        self.metadata = loaded_metadata.copy()
        self.metadata[self.image_path_column] = self.metadata[self.image_path_column].map(
            self._resolve_image_path
        )

        if validate_paths:
            missing_paths = [
                str(image_path)
                for image_path in self.metadata[self.image_path_column].map(Path)
                if not image_path.exists()
            ]
            if missing_paths:
                preview = ", ".join(missing_paths[:5])
                raise FileNotFoundError(
                    "Metadata references missing image files under "
                    f"{self.image_root}. Examples: {preview}"
                )

    def _resolve_image_path(self, image_path_value: object) -> str:
        """Resolve one image path against the configured image root if needed."""

        image_path = Path(str(image_path_value))
        if image_path.is_absolute():
            return str(image_path)
        return str(self.image_root / image_path)

    def __len__(self) -> int:
        """Return the number of rows in the metadata table."""

        return len(self.metadata)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Load one image sample and associated metadata."""

        row = self.metadata.iloc[index]
        image_path = Path(str(row[self.image_path_column]))

        with Image.open(image_path) as image:
            image_rgb = image.convert("RGB")

        transformed_image = self.transform(image_rgb) if self.transform is not None else image_rgb

        return {
            "image": transformed_image,
            "label": int(row[self.label_column]),
            "image_path": str(image_path),
            "id_code": str(row[self.id_column]),
        }
