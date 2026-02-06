"""Storage utilities for Forge results."""

from pathlib import Path
from django.conf import settings

# Results directory
RESULTS_DIR = Path(settings.BASE_DIR) / "forge_results"
RESULTS_DIR.mkdir(exist_ok=True)


def get_job_content(job) -> str:
    """Get content for a completed job."""
    if job.result_path.startswith("local:"):
        # Local file storage
        filename = job.result_path.replace("local:", "")
        file_path = RESULTS_DIR / filename
    else:
        file_path = Path(job.result_path)

    if file_path.exists():
        return file_path.read_text()

    raise FileNotFoundError(f"Result file not found: {job.result_path}")


def store_job_content(job_id: str, content: str, output_format: str) -> str:
    """Store job content and return path."""
    file_path = RESULTS_DIR / f"{job_id}.{output_format}"
    file_path.write_text(content)
    return str(file_path)


def delete_job_content(job) -> None:
    """Delete job content."""
    if job.result_path:
        file_path = Path(job.result_path)
        if file_path.exists():
            file_path.unlink()
