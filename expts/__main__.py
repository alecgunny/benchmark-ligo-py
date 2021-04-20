import argparse
import io
import logging
import os
import requests
import sys
import typing
from contextlib import contextmanager

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


class ServerMonitor:
    def __init__(self, cluster):
        server_monitor_node_pool = cloud.container.NodePool(
            name="server-monitor-pool",
            initial_node_count=1,
            config=cloud.gke.container.NodeConfig(
                machine_type="n1-standard-4",
                oauth_scopes=cloud.gke.OAUTH_SCOPES,
                labels={"triton-server-monitor": "true"}
            )
        )

        with cluster.manage_resource(
            server_monitor_node_pool, keep=True
        ) as server_monitor_node_pool:
            pass

        with cloud.deploy_file(
            os.path.join("apps", "triton-server-monitor.yaml"),
            ignore_if_exists=True
        ) as f:
            cluster.deploy(f)
        cluster.k8s_client.wait_for_deployment("triton-server-monitor")

        self.ip = cluster.k8s_client.wait_for_service(
            "triton-server-monitor"
        )

    @contextmanager
    def monitor(self, ip, model_name):
        response = requests.get(
            f"http://{self.ip}:5000/start",
            params={"ip": ip, "model-name": model_name}
        )
        response.raise_for_status()

        df = pd.DataFrame()
        try:
            yield df
        finally:
            response = requests.get(f"http://{self.ip}:5000/stop")
            response.raise_for_status()

        _df = pd.read_csv(io.BytesIO(response.content))
        for column in _df.columns:
            df[column] = _df[column]


def reset_server(
    expt,
    cluster,
    repo,
    vcpus_per_gpu
):
    """
    since Triton can't dynamically detect changes
    to the instance group without explicit model
    control, the simplest thing to do will be to
    spin up a new server instance each time our
    configuration changes
    """
    # set some values that we'll use to parse the deployment yaml
    deploy_file = os.path.join("apps", "triton-server.yaml")
    deploy_values = {
        "_file": os.path.join("apps", "values.yaml"),
        "repo": "gs://" + repo.bucket_name,
    }

    # start by updating all the model configs if
    # the instances-per-gpu have changed
    repo.update_model_configs_for_expt(expt)

    # now spin down the old deployment
    cluster.remove_deployment("tritonserver")

    # now add the new configuration details to our
    # yaml parsing values map
    max_cpus = 4 * vcpus_per_gpu
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
    return ip


def run_expt(expt, server_ip, server_monitor):
    max_exceptions, exceptions_this_expt = 5, 0

    server_url = f"{server_ip}:8001"
    str_kernel_size = f"{expt.kernel_stride:0.4f}".strip("0")
    model_name = f"kernel-stride-0{str_kernel_size}_gwe2e"

    generation_rate = 700
    last_client_df, last_server_df = None, None
    while True:
        logging.info(
            f"Benchmarking server with generation rate {generation_rate}"
        )

        with server_monitor.monitor(server_ip, model_name) as server_df:
            try:
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
            except Exception as e:
                exceptions_this_expt += 1
                if exceptions_this_expt == max_exceptions:
                    logging.error("Exception limited violated")
                    raise
                logging.warning("Encountered exception:")
                logging.warning(str(e))
                logging.warning("Retrying")

        client_stream.seek(0)
        client_df = pd.read_csv(client_stream)

        latency = client_df.request_return - client_df.message_start
        if np.percentile(latency, 99) < 0.1:
            generation_rate += 20
            last_client_df = client_df
            last_server_df = server_df
        else:
            logging.info(
                f"Experiment {expt} violated latency constraint "
                f"at generation rate {generation_rate}"
            )
            logging.info(
                "Latency percentiles: 50={} us, 95={} us".format(
                    int(np.percentile(latency, 50) * 10**6),
                    int(np.percentile(latency, 95) * 10**6)
                )
            )
            for model, d in server_df.groupby("model"):
                if "gwe2e" in model:
                    continue

                logging.info(
                    "Model {} queue percentiles: 50={} us, 95={} us".format(
                        model,
                        int(np.percentile(d.queue, 50)),
                        int(np.percentile(d.queue, 95))
                    )
                )

            if np.percentile(server_df.queue, 95) < 20000:
                logging.warning("Queue times stable, retrying")
            elif last_client_df is None:
                raise RuntimeError(
                    f"Expt {expt} failed before recording metrics"
                )
            else:
                break

    return last_client_df, last_server_df, client_df, server_df


def run_inference_experiments(
    cluster: cloud.gke.Cluster,
    repo: cloud.GCSModelRepo,
    vcpus_per_gpu: int = 16,
    keep: bool = False,
    experiment_interval: float = 40.0,
):
    server_monitor = ServerMonitor(cluster)

    # configure the server node pool
    max_cpus = 4 * vcpus_per_gpu
    node_pool = cloud.container.NodePool(
        name="triton-t4-pool",
        initial_node_count=1,
        config=cloud.gke.t4_node_config(vcpus=max_cpus, gpus=4)
    )

    # spin up the node pool on the cluster
    with cluster.manage_resource(node_pool, keep=keep) as node_pool:
        # make sure NVIDIA drivers got installed
        deploy_gpu_drivers(cluster)
        cluster.k8s_client.wait_for_daemon_set(name="nvidia-driver-installer")

        # iterate through our experiments and collect the results
        results = [[] for _ in range(4)]
        fnames = [
            "client-results.csv",
            "server-results.csv",
            "unstable-client-results.csv",
            "unstable-server-results.csv"
        ]

        current_instances, current_gpus = 0, 0
        try:
            for expt in expts:
                logging.info(f"Running expt {expt}")
                if (
                    current_instances != expt.instances or
                    current_gpus != expt.gpus
                ):
                    server_ip = reset_server(
                        expt, cluster, repo, vcpus_per_gpu
                    )

                dfs = run_expt(expt, server_ip, server_monitor)
                for result, df in zip(results, dfs):
                    df["kernel_stride"] = expt.kernel_stride
                    df["instances"] = expt.instances
                    df["gpus"] = expt.gpus
                    result.append(df)
        finally:
            for result, fname in zip(results, fnames):
                df = pd.concat(result, axis=0, ignore_index=True)
                df.to_csv(fname, index=False)


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
        run_inference_experiments(cluster, repo, vcpus_per_gpu=16, keep=keep)
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

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    manager = main(**vars(flags))
