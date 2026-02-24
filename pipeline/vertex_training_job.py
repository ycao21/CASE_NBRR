import argparse
import os
import pathlib
import re
import shutil
import subprocess
import tempfile
from typing import Tuple

import yaml
from google.cloud import aiplatform, storage


def get_latest_config_version(
    bucket_name: str, prefix: str, package_name: str, package_version: str
) -> int:
    """
    Get the latest config version from GCS by listing all matching files and finding max version.

    Args:
        bucket_name: GCS bucket name
        prefix: Path prefix in bucket
        package_name: Name of the package
        package_version: Version of the package

    Returns:
        Latest version number found, or 0 if no existing configs
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Convert package version format for matching
    package_version = package_version.replace(".", "_")

    # Create regex pattern to match config files and extract version
    pattern = f"config-{package_name}-{package_version}-v(\\d+)\\.yaml$"

    latest_version = 0
    # List all blobs in the prefix path
    blobs = bucket.list_blobs(prefix=prefix)

    for blob in blobs:
        match = re.search(pattern, blob.name)
        if match:
            version = int(match.group(1))
            latest_version = max(latest_version, version)

    return latest_version


def parse_gcs_uri(gcs_uri: str) -> Tuple[str, str]:
    """
    Parse GCS URI into bucket name and blob path.

    Args:
        gcs_uri: Full GCS URI (e.g., 'gs://bucket-name/path/to/file')

    Returns:
        Tuple of (bucket_name, blob_path)
    """
    parts = gcs_uri.replace("gs://", "").split("/", 1)
    return parts[0], parts[1] if len(parts) > 1 else ""


def upload_file_to_gcs(local_file_path: str, gcs_uri: str) -> None:
    """
    Upload a local file to GCS.

    Args:
        local_file_path: Path to local file
        gcs_uri: Destination GCS URI
    """
    client = storage.Client()
    bucket_name, blob_path = parse_gcs_uri(gcs_uri)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_file_path)
    print(f"Uploaded {local_file_path} to {gcs_uri}")


def upload_config_with_auto_version(
    local_config_path: str,
    package_name: str,
    package_version: str,
    base_gcs_path: str,
) -> str:
    """
    Upload config file with automatically incremented version.

    Args:
        local_config_path: Path to local config file
        package_name: Name of the package
        package_version: Version of the package
        base_gcs_path: Base GCS path for configs (e.g., 'gs://bucket/path/to/configs')

    Returns:
        The full GCS URI where the file was uploaded
    """
    # Parse base GCS path
    bucket_name, prefix = parse_gcs_uri(base_gcs_path)

    # Get latest version and increment
    latest_version = get_latest_config_version(
        bucket_name, prefix, package_name, package_version
    )
    new_version = latest_version + 1

    # Format package version for filename
    package_version_formatted = package_version.replace(".", "_")

    # Construct new config filename
    config_filename = (
        f"config-{package_name}-{package_version_formatted}-v{new_version}.yaml"
    )

    # Construct full GCS URI
    full_gcs_uri = f"{base_gcs_path.rstrip('/')}/{config_filename}"

    # Upload file
    upload_file_to_gcs(local_config_path, full_gcs_uri)

    return full_gcs_uri


def build_and_upload_package(project_root: str, staging_bucket: str, package_name: str, package_version: str) -> str:
    """Build source distribution package and upload to GCS"""
    print("=" * 60)
    print("Building Python package...")
    print("=" * 60)
    
    # Create a temporary directory for the package
    with tempfile.TemporaryDirectory() as tmpdir:
        pkg_dir = os.path.join(tmpdir, package_name)
        
        # Copy src/ directory contents directly into package directory
        # This makes the package structure: package_name/*.py instead of package_name/src/*.py
        src_dir = os.path.join(project_root, "src")
        
        if os.path.exists(src_dir):
            shutil.copytree(src_dir, pkg_dir)
            print(f"Copied src/ contents to {package_name}/ package")
        else:
            raise FileNotFoundError(f"src directory not found at {src_dir}")
        
        # Note: configs/ and pipeline/ are NOT copied to the package
        # Both configs are uploaded to GCS and passed as GCS URIs to the runner
        
        # Create setup.py (minimal, like XGBoost - no README needed)
        setup_content = f'''
from setuptools import setup, find_packages

setup(
    name="{package_name}",
    version="{package_version}",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "torch>=2.0.0",
        "pyarrow>=14.0.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "loguru>=0.7.0",
        "google-cloud-storage>=2.10.0",
        "tqdm>=4.65.0",
        "tensorboard>=2.20.0",
    ],
)
'''
        with open(os.path.join(tmpdir, "setup.py"), "w") as f:
            f.write(setup_content)
        
        # Build source distribution
        print(f"Building source distribution...")
        subprocess.run(
            ["python", "setup.py", "sdist", "--formats=gztar"],
            cwd=tmpdir,
            check=True
        )
        
        # Find the built tarball
        dist_dir = os.path.join(tmpdir, "dist")
        tarballs = [f for f in os.listdir(dist_dir) if f.endswith(".tar.gz")]
        if not tarballs:
            raise FileNotFoundError(f"No tar.gz file found in {dist_dir}")
        
        tarball = tarballs[0]
        tarball_path = os.path.join(dist_dir, tarball)
        print(f"Built package: {tarball}")
        
        # Upload to GCS
        print("Uploading package to GCS...")
        bucket_name = staging_bucket.replace("gs://", "").split("/")[0]
        blob_path = "/".join(staging_bucket.replace("gs://", "").split("/")[1:]) + f"/training-job-dist/{tarball}"
        
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        blob.upload_from_filename(tarball_path)
        
        gcs_package_uri = f"gs://{bucket_name}/{blob_path}"
        print(f"Package uploaded to: {gcs_package_uri}")
        print("=" * 60)
        
        return gcs_package_uri


def run(config_file, train_config, **kwargs):
    """
    Submit training runner job to Vertex AI.
    
    Args:
        config_file: Path to vertex.yaml config file
        train_config: Path to inference.yaml config file
        **kwargs: Additional overrides
    """
    # Set credentials to the service account with Vertex AI permissions
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/u/users/svc-p13nexplore/vertexai_credentials/explore_sa.json"
    gcs_creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    print(f"GCS Creds Path: {gcs_creds}")
    
    config_file_path = str(pathlib.Path(__file__).parent.parent / config_file)
    train_config_path = str(pathlib.Path(__file__).parent.parent / train_config)
    
    with open(config_file_path, "r") as f:
        vertex_config = yaml.safe_load(f)["vertex"]
    
    package_name = vertex_config["package_name"]
    package_version = vertex_config["package_version"]
    package_version_label = package_version.replace(".", "_")
    staging_bucket = vertex_config["staging_bucket"]
    run_type = vertex_config.get("run_type", "training")
    service_account = vertex_config["service_account"]
    network = vertex_config["network"]
    machine_type = vertex_config["machine_type"]
    replica_count = vertex_config.get("replica_count", 1)
    accelerator_type = vertex_config["accelerator_type"]
    accelerator_count = vertex_config["accelerator_count"]
    container_uri = vertex_config["container_uri"]
    location = vertex_config["region"]
    project_id = vertex_config["project_id"]
    module_path = vertex_config.get("module_path", "runner")
    
    # Upload training config with auto-versioning
    train_config_uri = upload_config_with_auto_version(
        local_config_path=train_config_path,
        package_name=package_name,
        package_version=package_version,
        base_gcs_path=f"{staging_bucket}/training-job-config",
    )
    print(f"Uploaded training config to {train_config_uri}")
    
    # Upload vertex config to GCS (infrastructure settings)
    vertex_config_gcs = f"{staging_bucket}/training-job-config/vertex.yaml"
    upload_file_to_gcs(config_file_path, vertex_config_gcs)
    print(f"Uploaded vertex config to {vertex_config_gcs}")
    
    # Build and upload Python package
    project_root = str(pathlib.Path(__file__).parent.parent)
    python_package_gcs_uri = build_and_upload_package(
        project_root=project_root,
        staging_bucket=staging_bucket,
        package_name=package_name,
        package_version=package_version
    )
    vertex_staging_bucket = f"{staging_bucket}/training-job-output"
    print(f"Vertex Staging bucket for job outputs: {vertex_staging_bucket}")
    job_name = vertex_config.get("job_name", "set-pb-training-job")
    
    aiplatform.init(project=project_id, location=location)
    job = aiplatform.CustomPythonPackageTrainingJob(
        display_name=job_name,
        python_package_gcs_uri=python_package_gcs_uri,
        python_module_name=f"{package_name}.{module_path}",  # src.runner
        container_uri=container_uri,
        staging_bucket=vertex_staging_bucket,
        labels={"package_version": package_version_label, "run_type": run_type},
    )
    
    # Prepare runner arguments
    # Both configs are uploaded to GCS and passed as GCS URIs
    runner_args = [
        "--config", train_config_uri, 
        "--vertex_config", vertex_config_gcs
    ]
    
    # Add optional overrides
    for key, value in kwargs.items():
        if value is not None:
            runner_args.extend([f"--{key}", str(value)])
    
    job.run(
        args=runner_args,
        replica_count=replica_count,
        machine_type=machine_type,
        accelerator_type=accelerator_type,
        accelerator_count=accelerator_count,
        service_account=service_account,
        network=network,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Repurchase Model Training Runner")
    parser.add_argument(
        "--config_file",
        type=str,
        default="pipeline/vertex_train.yaml",
        help="Path to Vertex AI config (default: pipeline/vertex_train.yaml)"
    )
    parser.add_argument(
        "--train_config",
        type=str,
        default="configs/train.yaml",
        help="Path to training config"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        help="Override batch size"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        help="Override epochs"
    )
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        help="Override learning rate"
    )
    
    args = parser.parse_args()
    run(**vars(args))
