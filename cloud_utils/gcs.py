import os
import re
import typing
from functools import lru_cache

from google.cloud import storage
from google.oauth2 import service_account

from cloud_utils.utils import clear_repo


class GCSModelRepo:
    def __init__(
        self,
        bucket_name: str,
        credentials: typing.Optional[service_account.Credentials] = None
    ):
        self.bucket_name = bucket_name
        self._client = storage.client.Client(credentials=credentials)

    @property
    @lru_cache(None)
    def bucket(self):
        try:
            return self._client.get_bucket(self.bucket_name)
        except Exception as e:
            try:
                if e.code == 404:
                    try:
                        return self._client.create_bucket(self.bucket_name)
                    except Exception as e:
                        # TODO: some check on whether this already exists
                        raise e
                else:
                    raise
            except AttributeError:
                raise e

    def export_repo(
        self,
        repo_dir: str,
        start_fresh: bool = True,
        clear: bool = True
    ):
        if start_fresh:
            for blob in self._client.list_blobs(self.bucket):
                blob.delete()

        for root, _, files in os.walk(repo_dir):
            for f in files:
                path = os.path.join(root, f)

                # get rid of root level path and replace
                # path separaters in case we're on Windows
                blob_path = path.replace(os.path.join(repo_dir, ""), "").replace(
                    "\\", "/"
                )
                print(f"Copying {path} to {blob_path}")

                blob = self.bucket.blob(blob_path)
                blob.upload_from_filename(path)

        if clear:
            clear_repo(repo_dir)

    def update_model_configs_for_expt(self, expt):
        """
        TODO: in a more general framework, get rid of this
        """
        new_blobs = {}
        for blob in self._client.list_blobs(self.bucket):
            if (
                blob.name.startswith(f"kernel-stride-{expt.kernel_stride:0.3f}")
                and blob.name.endswith(".pbtxt")
                and "snapshotter" not in blob.name
            ):
                content = blob.download_as_bytes().decode("utf-8")
                blob_name = blob.name
                blob.delete()

                content = re.sub(
                    "(?<=count: )[0-9]", str(expt.instances), content
                )
                new_blobs[blob_name] = content.encode("utf-8")

        for blob_name, content in new_blobs.items():
            blob = self.bucket.blob(blob_name)
            blob.upload_from_string(
                content,
                content_type="application/octet-stream"
            )
