import io


def upload_to_gcloud_bucket(project: str, bucket: str, location: str, fname: str):
    """
    Upload local file to Google Cloud bucket.
    """
    # if TYPE_CHECKING:
    from google.cloud import storage

    client = storage.Client(project=project)
    bucket = client.get_bucket(bucket)

    with open(fname, 'rb') as f:
        content = io.BytesIO(f.read())

    tensor = bucket.blob(location + '/' + fname)
    tensor.upload_from_file(content, timeout=7200)
