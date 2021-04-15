import os
import re
import requests
import time
import typing
import yaml
from base64 import b64decode
from contextlib import contextmanager
from tempfile import NamedTemporaryFile

import kubernetes

from cloud_utils.utils import wait_for


class K8sApiClient:
    def __init__(self, cluster):
        try:
            response = cluster.get()
        except Exception as e:
            try:
                if e.code == 404:
                    raise RuntimeError(
                        f"Cluster {cluster.name} not currently deployed"
                    )
                raise
            except AttributeError:
                raise e

        # create configuration using bare minimum info
        configuration = kubernetes.client.Configuration()
        configuration.host = f"https://{response.endpoint}"
        with NamedTemporaryFile(delete=False) as ca_cert:
            ca_cert.write(
                b64decode(response.master_auth.cluster_ca_certificate)
            )
        configuration.ssl_ca_cert = ca_cert.name
        configuration.api_key_prefix["authorization"] = "Bearer"

        # get credentials for conencting to server
        # GCP code lifted from
        # https://cloud.google.com/compute/docs/access/
        # create-enable-service-accounts-for-instances#applications
        if cluster.parent.credentials is None:
            METADATA_URL = "http://metadata.google.internal/computeMetadata/v1"
            METADATA_HEADERS = {"Metadata-Flavor": "Google"}
            SERVICE_ACCOUNT = "default"

            url = "{}/instance/service-accounts/{}/token".format(
                METADATA_URL, SERVICE_ACCOUNT)

            # Request an access token from the metadata server.
            r = requests.get(url, headers=METADATA_HEADERS)
            r.raise_for_status()

            # Extract the access token from the response.
            access_token = r.json()["access_token"]
        else:
            access_token = cluster.parent.credentials.token

        configuration.api_key["authorization"] = access_token

        # return client instantiated with configuration
        self._client = kubernetes.client.ApiClient(configuration)

    def create_from_yaml(self, file: str):
        return kubernetes.utils.create_from_yaml(self._client, file)

    def remove_deployment(self, name: str, namespace: str = "default"):
        app_client = kubernetes.client.AppsV1Api(self._client)

        def _try_cmd(cmd):
            try:
                cmd(name=name, namespace=namespace)
            except Exception as e:
                try:
                    body = yaml.safe_load(e.body)
                    if body["code"] == 404:
                        return True
                    else:
                        return False
                except AttributeError:
                    raise e
                except Exception:
                    raise
            return False

        _try_cmd(app_client.delete_namespaced_deployment)

        def _deleted_callback():
            return _try_cmd(app_client.read_namespaced_deployment)

        wait_for(
            _deleted_callback,
            f"Waiting for deployment {name} to delete",
            f"Deployment {name} deleted"
        )

    def wait_for_deployment(self, name: str, namespace: str = "default"):
        app_client = kubernetes.client.AppsV1Api(self._client)

        _start_time = time.time()
        _grace_period_seconds = 10

        def _ready_callback():
            try:
                response = app_client.read_namespaced_deployment_status(
                    name=name, namespace=namespace
                )
            except kubernetes.client.ApiException:
                raise RuntimeError(
                    f"Deployment {name} no longer exists!"
                )
            conditions = response.status.conditions
            if conditions is None:
                return False
            statuses = {i.type: eval(i.status) for i in conditions}

            try:
                if statuses["Available"]:
                    return True
            except KeyError:
                if (time.time() - _start_time) > _grace_period_seconds:
                    raise ValueError("Couldn't find readiness status")

            try:
                if not statuses["Progressing"]:
                    raise RuntimeError(
                        f"Deployment {name} stopped progressing"
                    )
            except KeyError:
                pass
            finally:
                return False

        wait_for(
            _ready_callback,
            f"Waiting for deployment {name} to deploy",
            f"Deployment {name} ready"
        )

    def wait_for_service(self, name: str, namespace: str = "default"):
        core_client = kubernetes.client.CoreV1Api(self._client)

        def _ready_callback():
            try:
                response = core_client.read_namespaced_service_status(
                    name=name, namespace=namespace
                )
            except kubernetes.client.ApiException:
                raise RuntimeError(
                    f"Service {name} no longer exists!"
                )

            ip = response.status.load_balancer.ingress[0].ip
            return ip or False

        return wait_for(
            _ready_callback,
            f"Waiting for service {name} to be ready",
            f"Service {name} ready"
        )

    def wait_for_daemon_set(self, name: str, namespace: str = "kube-system"):
        core_client = kubernetes.client.CoreV1Api(self._client)

        def _ready_callback():
            try:
                response = core_client.read_namespaced_daemon_set_status(
                    name=name, namespace=namespace
                )
            except kubernetes.client.ApiException:
                raise RuntimeError(
                    f"Daemon set {name} no longer exists!"
                )

            status = response.status
            return (
                status.desired_number_scheduled == status.number_ready
            )


@contextmanager
def deploy_file(
    file: str,
    repo: typing.Optional[str] = None,
    branch: typing.Optional[str] = None,
    values: typing.Optional[typing.Dict[str, str]] = None,
    ignore_if_exists: bool = True
):
    if repo is not None:
        if branch is None:
            branches = ["main", "master"]
        else:
            branches = [branch]

        for branch in branches:
            url_header = "https://raw.githubusercontent.com"
            url = f"{url_header}/{repo}/{branch}/{file}"

            try:
                yaml_content = requests.get(url).content.decode("utf-8")
            except Exception:
                pass
            else:
                break
        else:
            raise ValueError(
                f"Couldn't find file {file} at github repo {repo}, ",
                "tried looking in branches {}".format(
                    ", ".join(branches)
                )
            )
    else:
        with open(file, "r") as f:
            yaml_content = f.read()

    values = values or {}
    values = values.copy()
    try:
        # try to load in values from file
        values_file = values.pop("_file")
    except KeyError:
        pass
    else:
        # use explicitly passed values to overwrite
        # values in file
        with open(values_file, "r") as f:
            values_map = yaml.safe_load(f)
        values_map.update(values)
        values = values_map

    # look for any Go variable indicators and try to
    # fill them in with their value from `values`
    def replace_fn(match):
        varname = re.search(
            "(?<={{ .Values.)[a-zA-Z0-9]+?(?= }})", match.group(0)
        ).group(0)
        try:
            return str(values[varname])
        except KeyError:
            raise ValueError(
                "No value provided for wildcard {}".format(
                    varname
                )
            )
    yaml_content = re.sub(
        "{{ .Values.[a-zA-Z0-9]+? }}", replace_fn, yaml_content
    )

    # write formatted yaml to temporary file
    with NamedTemporaryFile(mode="w", delete=False) as f:
        f.write(yaml_content)
        file = f.name

    try:
        try:
            yield file
        except kubernetes.utils.FailToCreateError as e:
            if not ignore_if_exists:
                # doesn't matter what the issue was,
                # delete the temp files and raise
                raise

            # try to load api exception information
            for exc in e.api_exceptions:
                info = yaml.safe_load(exc.body)
                if info["reason"] != "AlreadyExists":
                    raise
    finally:
        # remove the temporary file no matter
        # what happens
        os.remove(file)
