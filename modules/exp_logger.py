import csv
import os
from typing import Any, Dict, List, Optional, Set

import yaml
from loguru import logger


class ExpLogger:
    """
    Logs data to the local file system in CSV and YAML formats.

    Version folders are now named:
        version_{number}_{attack_level}_{dataset}
    e.g.:
        version_18_attack1_low_news
        version_19_attack2_mid_medical
        version_20_attack3_high_mmlu

    If no attack_label or dataset_label is provided, falls back to:
        version_{number}
    """

    def __init__(
        self,
        root_dir: str = "./",
        log_dir_name: str = "logs",
        attack_label: Optional[str] = None,
        dataset_label: Optional[str] = None,
    ):
        super().__init__()
        self._root_dir = os.path.abspath(root_dir)
        self._log_dir_name = log_dir_name
        self._attack_label = attack_label    # e.g. "attack1_low"
        self._dataset_label = dataset_label  # e.g. "news"
        self._version = None
        self._csv_loggers: Dict[str, "_CSVWriter"] = {}
        self._yaml_loggers: Dict[str, "_YAMLWriter"] = {}

    @property
    def version(self) -> int:
        """Gets the experiment version number (auto-incremented)."""
        if self._version is None:
            self._version = self._get_next_version()
        return self._version

    @property
    def experiment_dir(self) -> str:
        """
        The directory path for this experiment's logs.

        Format: version_{number}_{attack_label}_{dataset_label}
        Example: version_18_attack1_low_news
        """
        # Build the suffix from optional labels
        suffix_parts = []
        if self._attack_label:
            suffix_parts.append(self._attack_label)
        if self._dataset_label:
            suffix_parts.append(self._dataset_label)

        if suffix_parts:
            version_name = f"version_{self.version}_{'_'.join(suffix_parts)}"
        else:
            version_name = f"version_{self.version}"

        return os.path.join(self._root_dir, self._log_dir_name, version_name)

    def get_csv_logger(self, logger_name: str) -> "_CSVWriter":
        if logger_name not in self._csv_loggers:
            os.makedirs(self.experiment_dir, exist_ok=True)
            self._csv_loggers[logger_name] = _CSVWriter(
                log_dir=self.experiment_dir, logger_name=logger_name
            )
        return self._csv_loggers[logger_name]

    def get_yaml_logger(self, logger_name: str) -> "_YAMLWriter":
        if logger_name not in self._yaml_loggers:
            os.makedirs(self.experiment_dir, exist_ok=True)
            self._yaml_loggers[logger_name] = _YAMLWriter(
                log_dir=self.experiment_dir, logger_name=logger_name
            )
        return self._yaml_loggers[logger_name]

    def _get_next_version(self) -> int:
        """
        Determines the next available version number by scanning existing folders.

        Folders can now be named version_18 OR version_18_attack1_low_news.
        Both are handled by splitting on "_" and reading only the second token.
        """
        experiment_root = os.path.join(self._root_dir, self._log_dir_name)

        if not os.path.isdir(experiment_root):
            return 0

        existing_versions = []
        for dir_name in os.listdir(experiment_root):
            full_path = os.path.join(experiment_root, dir_name)
            if os.path.isdir(full_path) and dir_name.startswith("version_"):
                try:
                    # Works for both "version_18" and "version_18_attack1_low_news"
                    # because we always take index [1] which is the number part
                    version_num = int(dir_name.split("_")[1])
                    existing_versions.append(version_num)
                except (ValueError, IndexError):
                    pass

        return max(existing_versions, default=-1) + 1


# ---------------------------------------------------------------------------
# Internal helpers — unchanged from original
# ---------------------------------------------------------------------------

class _CSVWriter:
    """CSV writer for ExpLogger."""

    def __init__(self, log_dir: str, logger_name: str) -> None:
        self.data_buffer: List[Dict[str, float]] = []
        self.fieldnames: List[str] = []
        self.log_dir = log_dir
        self.log_file_name = f"{logger_name}.csv"
        self.log_file_path = os.path.join(self.log_dir, self.log_file_name)

    def log(self, data_dict: Dict[str, Any]) -> None:
        self.data_buffer.append(data_dict)

    def save(self) -> None:
        if not self.data_buffer:
            logger.warning(f"No data to save for {self.log_file_name}.")
            return

        new_fieldnames = self._update_fieldnames()
        file_exists = os.path.isfile(self.log_file_path)

        if new_fieldnames and file_exists:
            self._rewrite_csv_with_new_header(self.fieldnames)

        with open(self.log_file_path, mode=("a" if file_exists else "w"),
                  errors="surrogatepass", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames, escapechar="\\")
            if not file_exists:
                writer.writeheader()
            writer.writerows(self.data_buffer)

        self.data_buffer = []

    def _update_fieldnames(self) -> Set[str]:
        current_fieldnames = set().union(*self.data_buffer)
        new_fieldnames = current_fieldnames - set(self.fieldnames)
        self.fieldnames.extend(new_fieldnames)
        self.fieldnames.sort()
        return new_fieldnames

    def _rewrite_csv_with_new_header(self, fieldnames: List[str]) -> None:
        with open(self.log_file_path, "r", errors="surrogatepass", newline="") as csvfile:
            reader = csv.DictReader(csvfile, escapechar="\\")
            original_data = list(reader)

        with open(self.log_file_path, "w", errors="surrogatepass", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, escapechar="\\")
            writer.writeheader()
            writer.writerows(original_data)


class _YAMLWriter:
    """YAML writer for ExpLogger."""

    def __init__(self, log_dir: str, logger_name: str) -> None:
        self.data_buffer: Dict[str, Any] = {}
        self.log_dir = log_dir
        self.log_file_name = f"{logger_name}.yaml"
        self.log_file_path = os.path.join(self.log_dir, self.log_file_name)

    def log(self, data_dict: Dict[str, Any]) -> None:
        self.data_buffer.update(data_dict)

    def save(self) -> None:
        if not self.data_buffer:
            logger.warning(f"No data to save for {self.log_file_name}.")
            return

        with open(self.log_file_path, "w") as yamlfile:
            yaml.dump(self.data_buffer, yamlfile, default_flow_style=False)

        self.data_buffer = {}