import os
from pathlib import Path
import pooch

# ---- Configuration ----
DATA_PACKAGE_NAME ="robokudo_test_data"
DATA_VERSION = "b991b2f15cd734672f449349a9e566fb67aad81e"
KNOWN_HASH = "sha256:b9150798870b7e7d067387dc295e661a55e2360c8c9cd944b6815d0ec59047e5"

URL = (
    "https://gitlab.informatik.uni-bremen.de/robokudo/robokudo_test_data/-/jobs/artifacts/"
    f"{DATA_VERSION}/raw/robokudo_test_data-{DATA_VERSION}.zip?job=package_zip"
)

FILENAME = f"{DATA_PACKAGE_NAME}.zip"

def test_data_path() -> Path:
    """
    Will retrieve robokudo test data from the CI, if not locally present.
    """

    downloader = pooch.HTTPDownloader()

    # Download + verify + unzip
    extracted_files = pooch.retrieve(
        url=f"{URL}",
        known_hash=KNOWN_HASH,
        path=pooch.os_cache(DATA_PACKAGE_NAME),  # ~/.cache/DATA_PACKAGE_NAME
        fname=FILENAME,
        downloader=downloader,
        processor=pooch.Unzip(),
    )

    # Pooch returns list of extracted file paths
    # We usually want the root extracted directory:
    extracted_dir = Path(extracted_files[0]).parents[0]

    return extracted_dir


if __name__ == "__main__":
    dataset_path = retrieve_test_data()
    print("Dataset available at:", dataset_path)
