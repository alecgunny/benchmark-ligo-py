import re
import time
import typing
from contextlib import contextmanager
from functools import partial

import attr
from google.auth.transport.requests import Request as AuthRequest
from google.oauth2 import service_account
from google.cloud import container_v1 as container

from cloud_utils.k8s import K8sApiClient
from cloud_utils.utils import wait_for


OAUTH_SCOPES = [
    "https://www.googleapis.com/auth/devstorage.read_only",
    "https://www.googleapis.com/auth/logging.write",
    "https://www.googleapis.com/auth/monitoring",
    "https://www.googleapis.com/auth/service.management.readonly",
    "https://www.googleapis.com/auth/servicecontrol",
    "https://www.googleapis.com/auth/trace.append"
]


def snakeify(name: str) -> str:
    return re.sub("(?<!^)(?=[A-Z])", "_", name).lower()


class ThrottledClient:
    def __init__(self, credentials=None, throttle_secs=1.0):
        self._client = container.ClusterManagerClient(credentials=credentials)
        self.throttle_secs = throttle_secs
        self._last_request_time = time.time()

    def make_request(self, request, **kwargs):
        request_fn_name = snakeify(
            type(request).__name__.replace("Request", "")
        )
        request_fn = getattr(self._client, request_fn_name)
        while (time.time() - self._last_request_time) < self.throttle_secs:
            time.sleep(0.01)
        return request_fn(request=request, **kwargs)


@attr.s(auto_attribs=True)
class Resource:
    _name: str
    parent: "Resource"

    @property
    def client(self):
        return self.parent.client

    @property
    def resource_type(self):
        return type(self).__name__

    @property
    def name(self):
        resource_type = self.resource_type
        camel = resource_type[0].lower() + resource_type[1:]
        return self.parent.name + "/{}/{}".format(camel, self._name)

    @classmethod
    def create(cls, resource, parent):
        resource_type = type(resource).__name__
        if resource_type == "Cluster":
            cls = Cluster
        elif resource_type == "NodePool":
            cls = NodePool
        else:
            raise TypeError(f"Unknown GKE resource type {resource_type}")

        obj = cls(resource.name, parent)
        create_request_cls = getattr(
            container, f"Create{obj.resource_type}Request"
        )

        resource_type = snakeify(obj.resource_type)
        kwargs = {
            resource_type: resource,
            "parent": parent.name
        }
        create_request = create_request_cls(**kwargs)
        try:
            obj.client.make_request(create_request)
        except Exception as e:
            try:
                if e.code != 409:
                    raise
            except AttributeError:
                raise e
        return obj

    def delete(self):
        delete_request_cls = getattr(
            container, f"Delete{self.resource_type}Request"
        )
        delete_request = delete_request_cls(name=self.name)
        return self.client.make_request(delete_request)

    def get(self, timeout=None):
        get_request_cls = getattr(
            container, f"Get{self.resource_type}Request"
        )
        get_request = get_request_cls(name=self.name)
        return self.client.make_request(get_request, timeout=timeout)


class NodePool(Resource):
    pass


class Cluster(Resource):
    def __init__(self, resource, parent):
        super().__init__(resource, parent)
        self._k8s_client = None

    def _make_k8s_client(self):
        if self._k8s_client is not None:
            raise ValueError(
                f"Already created kubernetes client for cluster {self.name}"
            )
        self._k8s_client = K8sApiClient(self)

    @property
    def k8s_client(self):
        if self._k8s_client is None:
            self._make_k8s_client()
        return self._k8s_client

    @classmethod
    def create(cls, resource, parent):
        obj = super().create(resource, parent)
        try:
            obj._make_k8s_client()
        except ValueError:
            pass
        return obj

    def deploy(self, file: str):
        return self.k8s_client.create_from_yaml(file)

    def remove_deployment(self, name: str, namespace: str = "default"):
        return self.k8s_client.remove_deployment(name, namespace)


def resource_ready_callback(resource):
    try:
        status = resource.get(timeout=5).status
    except Exception:
        # TODO: something to catch here?
        raise
    if status == 2:
        return True
    elif status > 2:
        raise RuntimeError
    return False


def resource_delete_submit_callback(resource):
    # first try to submit the delete request,
    # possibly waiting for the resource to
    # become available to be deleted if we
    # need to
    try:
        resource.delete()
    except Exception as e:
        try:
            if e.code == 404:
                # resource is gone, we're good
                return True
            elif e.code != 400:
                # 400 means resource is tied up, so
                # wait and try again in a bit. Otherwise,
                # raise an error
                raise
            else:
                return False
        except AttributeError:
            # the exception didn't have a `.code`
            # attribute, so evidently something
            # else went wrong, raise it
            raise e
    else:
        # response went off ok, so we're good
        return True


def resource_delete_done_callback(resource):
    # now wait for the delete request to
    # be completed
    try:
        status = resource.get(timeout=5).status
    except Exception as e:
        try:
            if e.code == 404:
                # resource is gone, so we're good
                # to exit
                return True
            # some other error occured, raise it
            raise
        except AttributeError:
            # a non-HTTP error occurred, raise it
            raise e

    if status > 4:
        # something bad happened to the resource,
        # raise the issue
        raise RuntimeError(status)
    return False


class GKEClusterManager:
    def __init__(
        self,
        project: str,
        zone: str,
        credentials: typing.Optional[service_account.Credentials] = None
    ):
        self.credentials = credentials
        self.client = ThrottledClient(self.credentials)

        self.name = f"projects/{project}/locations/{zone}"
        self.resources = {}

    @contextmanager
    def manage_resource(self, resource, parent=None, keep=False):
        parent = parent or self
        resource = Resource.create(resource, parent)

        resource_type = snakeify(resource.resource_type).replace("_", " ")
        resource_msg = resource_type + " " + resource.name

        wait_for(
            partial(resource_ready_callback, resource),
            f"Waiting for {resource_msg} to become ready",
            f"{resource_msg} ready"
        )

        def delete_resource(raised):
            if keep:
                return
            elif raised:
                print(f"Encountered error, removing {resource_msg}")

            wait_for(
                partial(resource_delete_submit_callback, resource),
                f"Waiting for {resource_msg} to become available to delete",
                f"{resource_msg} delete request submitted"
            )

            wait_for(
                partial(resource_delete_done_callback, resource),
                f"Waiting for {resource_type} {resource.name} to delete",
                f"{resource_type} {resource.name} deleted"
            )
            self.resources.pop(resource.name)

        self.resources[resource.name] = resource
        raised = False
        try:
            yield resource
        except Exception:
            raised = True
            raise
        finally:
            delete_resource(raised)


def t4_node_config(vcpus=8, gpus=1, **kwargs):
    return container.NodeConfig(
        machine_type=f"n1-standard-{vcpus}",
        oauth_scopes=OAUTH_SCOPES,
        accelerators=[container.AcceleratorConfig(
            accelerator_count=gpus,
            accelerator_type="nvidia-tesla-t4"
        )],
        **kwargs
    )


def make_credentials(service_account_key_file):
    # use GKE credentials to create Kubernetes
    # configuration for cluster
    credentials = service_account.Credentials.from_service_account_file(
        service_account_key_file,
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    credentials.refresh(AuthRequest())
    return credentials
