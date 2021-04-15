import argparse
import io
import os
import requests
import typing

import pandas as pd
import numpy as np

import cloud_utils as cloud
from expt_configs import expts
from expts.offline import run_experiment


def deploy_gpu_drivers(cluster):
    # deploy GPU drivers daemonset on to cluster
    with cloud.deploy_file(
        "nvidia-driver-installer/cos/daemonset-preloaded.yaml",
        repo="GoogleCloudPlatform/container-engine-accelerators",
        branch="master",
        ignore_if_exists=True
    ) as f:
        cluster.deploy(f)


def run_inference_experiments(
    manager: cloud.GKEClusterManager,
    cluster: cloud.gke.Cluster,
    repo: cloud.GCSModelRepo,
    vcpus_per_gpu: int = 16,
    keep: bool = False,
    experiment_interval: float = 40.0,
):
    server_monitor_node_pool = cloud.container.NodePool(
        name="server-monitor-pool",
        initial_node_count=1,
        config=cloud.gke.container.NodeConfig(
            machine_type="n1-standard-4",
            oauth_scopes=cloud.gke.OAUTH_SCOPES,
            labels={"triton-server-monitor": "true"}
        )
    )

    with manager.manage_resource(
        server_monitor_node_pool, cluster, keep=True
    ) as server_monitor_node_pool:
        pass

    with cloud.deploy_file(
        os.path.join("apps", "triton-server-monitor.yaml"),
        ignore_if_exists=True
    ) as f:
        cluster.deploy(f)
    cluster.k8s_client.wait_for_deployment("triton-server-monitor")
    monitor_ip = cluster.k8s_client.wait_for_service(
        "triton-server-monitor"
    )

    # configure the server node pool
    max_cpus = 4 * vcpus_per_gpu
    node_pool = cloud.container.NodePool(
        name="triton-t4-pool",
        initial_node_count=1,
        config=cloud.gke.t4_node_config(vcpus=max_cpus, gpus=4)
    )

    # spin up the node pool on the cluster
    with manager.manage_resource(node_pool, cluster, keep=keep) as node_pool:
        # make sure NVIDIA drivers got installed
        deploy_gpu_drivers(cluster)
        cluster.k8s_client.wait_for_daemon_set(name="nvidia-driver-installer")

        # set some values that we'll use to parse the deployment yaml
        deploy_file = os.path.join("apps", "triton-server.yaml")
        deploy_values = {
            "_file": os.path.join("apps", "values.yaml"),
            "repo": "gs://" + repo.bucket_name,
        }

        # iterate through our experiments and collect the results
        client_results, server_results = [], []
        current_instances, current_gpus = 0, 0
        for expt in expts:
            if current_instances != expt.instances or current_gpus != expt.gpus:
                # since Triton can't dynamically detect changes
                # to the instance group without explicit model
                # control, the simplest thing to do will be to
                # spin up a new server instance each time our
                # configuration changes

                # start by updating all the model configs if
                # the instances-per-gpu have changed
                if current_instances != expt.instances:
                    repo.update_model_configs_for_expt(expt)

                # now spin down the old deployment
                cluster.remove_deployment("tritonserver")

                # now add the new configuration details to our
                # yaml parsing values map
                num_cpus = min(vcpus_per_gpu * expt.gpus, max_cpus - 1)
                deploy_values.update({"numGPUs": expt.gpus, "cpu": num_cpus})

                # deploy this new yaml onto the cluster
                with cloud.deploy_file(
                    deploy_file, values=deploy_values, ignore_if_exists=True
                ) as f:
                    cluster.deploy(f)

                # wait for it to be ready
                cluster.k8s_client.wait_for_deployment("tritonserver")
                ip = cluster.k8s_client.wait_for_service("tritonserver")

                current_instances = expt.instances
                current_gpus = expt.gpus

            generation_rate = 800
            last_client_df, last_server_df = None, None
            while True:
                server_url = f"{ip}:8001"
                model_name = f"kernel-stride-{expt.kernel_stride:0.4f}_gwe2e"
                requests.get(
                    f"http://{monitor_ip}:5000/start",
                    params={"url": server_url, "model-name": model_name}
                )

                client_stream = run_experiment(
                    url=server_url,
                    model_name=model_name,
                    model_version=1,
                    num_clients=1,
                    sequence_id=1001,
                    generation_rate=generation_rate,
                    num_iterations=50000,
                    warm_up=10,
                    filename=io.StringIO()
                )
                response = requests.get(f"http://{monitor_ip}:5000/stop")

                client_df = pd.read_csv(client_stream.getvalue())
                server_df = pd.read_csv(io.BytesIO(response.content))

                latency = client_df.request_return - client_df.message_start
                if np.percentile(latency, 99) < 0.1:
                    generation_rate += 20
                    last_client_df = client_df
                    last_server_df = server_df
                else:
                    break

            for df in [last_client_df, last_server_df]:
                df["kernel_stride"] = expt.kernel_stride
                df["instances"] = expt.instances
                df["gpus"] = expt.gpus
            client_results.append(last_client_df)
            server_results.append(last_server_df)

    client_results = pd.concat(client_results, axis=0, ignore_index=True)
    client_results.to_csv("client-results.csv", index=False)

    server_results = pd.concat(server_results, axis=0, ignore_index=True)
    server_results.to_csv("server-results.csv", index=False)


def main(
    project: str,
    cluster_name: str,
    zone: str = "us-west1-b",
    bucket: typing.Optional[str] = None,
    keep: bool = False
):
    manager = cloud.GKEClusterManager(project, zone)
    repo = cloud.GCSModelRepo(bucket or cluster_name + "_model-repo")

    cluster = cloud.container.Cluster(
        name=cluster_name,
        node_pools=[cloud.container.NodePool(
            name="default-pool",
            initial_node_count=2,
            config=cloud.container.NodeConfig()
        )]
    )

    with manager.manage_resource(cluster, keep=keep) as cluster:
        run_inference_experiments(
            manager, cluster, repo, vcpus_per_gpu=16, keep=True
        )
    return manager


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project",
        type=str,
        required=True
    )
    parser.add_argument(
        "--cluster-name",
        type=str,
        default="gw-benchmarking"
    )
    parser.add_argument(
        "--zone",
        type=str,
        default="us-west1-b"
    )
    parser.add_argument(
        "--bucket",
        type=str,
        default=None
    )
    parser.add_argument(
        "--keep",
        action="store_true",
    )
    flags = parser.parse_args()
    manager = main(**vars(flags))
