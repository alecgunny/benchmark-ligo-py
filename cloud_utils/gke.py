import re
import time
import typing
from contextlib import contextmanager

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
        self.credentials = credentials
        self._client = container.ClusterManagerClient(
            credentials=self.credentials
        )
        self.throttle_secs = throttle_secs
        self._last_request_time = time.time()

    @property
    def client(self):
        return self

    @property
    def name(self):
        return ""

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

    # TODO: start catching the specific HTTPException
    # subclass. Is it just the one from werkzeug?
    def is_ready(self):
        try:
            status = self.get(timeout=5).status
        except Exception:
            # TODO: something to catch here?
            raise
        if status == 2:
            return True
        elif status > 2:
            raise RuntimeError
        return False

    def submit_delete(self):
        # first try to submit the delete request,
        # possibly waiting for the resource to
        # become available to be deleted if we
        # need to
        try:
            self.delete()
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

    def is_deleted(self):
        # now wait for the delete request to
        # be completed
        try:
            status = self.get(timeout=5).status
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


class NodePool(Resource):
    pass


class ManagerResource(Resource):
    def __attrs_post_init__(self):
        self._resources = {}
        # TODO: include automatic resource detection here

    @property
    def managed_resource_type(self):
        raise NotImplementedError

    @property
    def resources(self):
        resources = self._resources.copy()
        for resource_name, resource in self._resources.items():
            try:
                subresources = resource.resources
            except AttributeError:
                continue
            for subname, subresource in subresources.items():
                resources[subname] = subresource
        return resources

    def _make_resource_message(resource):
        resource_type = snakeify(resource.resource_type).replace("_", " ")
        return resource_type + " " + resource.name

    def create_resource(self, resource):
        if type(resource).__name__ != self.managed_resource_type.__name__:
            raise TypeError("{} cannot manage resource {}".format(
                type(self).__name__, type(resource).__name__
            ))

        resource = Resource.create(resource, self)
        resource_msg = self._make_resource_message(resource)

        wait_for(
            resource.is_ready,
            f"Waiting for {resource_msg} to become ready",
            f"{resource_msg} ready"
        )
        self._resources[resource.name] = resource
        return resource

    def delete_resource(self, resource):
        resource_msg = self._make_resource_message(resource)

        wait_for(
            resource.submit_delete,
            f"Waiting for {resource_msg} to become available to delete",
            f"{resource_msg} delete request submitted"
        )

        wait_for(
            resource.is_deleted,
            f"Waiting for {resource_msg} to delete",
            f"{resource_msg} deleted"
        )
        self._resources.pop(resource.name)

    @contextmanager
    def manage_resource(self, resource, keep=False):
        resource = self.create_resource(resource)
        resource_msg = self._make_resource_message(resource)

        try:
            yield resource
        except Exception:
            if not keep:
                print(f"Encountered error, removing {resource_msg}")
            raise
        finally:
            if not keep:
                self.delete_resource(resource)


class Cluster(ManagerResource):
    def __attrs_post_init__(self):
        self._k8s_client = K8sApiClient(self)
        super().__attrs_post_init__()

    @property
    def managed_resource_type(self):
        return NodePool

    @property
    def k8s_client(self):
        return self._k8s_client

    def deploy(self, file: str):
        return self.k8s_client.create_from_yaml(file)

    def remove_deployment(self, name: str, namespace: str = "default"):
        return self.k8s_client.remove_deployment(name, namespace)


class GKEClusterManager(ManagerResource):
    def __init__(
        self,
        project: str,
        zone: str,
        credentials: typing.Optional[service_account.Credentials] = None
    ):
        parent = ThrottledClient(credentials)
        name = f"projects/{project}/locations/{zone}"
        super().__init__(name, parent)

    @property
    def managed_resource_type(self):
        return Cluster

    @property
    def name(self):
        return self._name


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
