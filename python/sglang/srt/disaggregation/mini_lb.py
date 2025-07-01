"""
Minimal HTTP load balancer for prefill and decode servers for testing.
"""

import asyncio
import random
import urllib
from itertools import chain
from typing import List, Dict, Any, Optional, Set, Union
from dataclasses import dataclass
import logging


from kubernetes import client, config
from kubernetes.client.rest import ApiException
from kubernetes.watch import Watch
from tenacity import retry, wait_exponential, stop_after_attempt

import aiohttp
import orjson
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import ORJSONResponse, Response, StreamingResponse
import threading
from threading import Thread

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger(__name__)


class PodInfo:
    """Pod information"""
    def __init__(self, name: str, ip: str, port: int, role_name: str,
                 roleset_name: str, service_name: str, role_replica_index: int,
                 bootstrap_port: int = 8998, roleset_index: Optional[int] = None):
        self.name = name
        self.ip = ip
        self.port = port
        self.role_name = role_name
        self.roleset_name = roleset_name
        self.service_name = service_name
        self.role_replica_index = role_replica_index
        self.bootstrap_port = bootstrap_port
        self.roleset_index = roleset_index

    @property
    def url(self) -> str:
        return f"http://{self.ip}:{self.port}"


class RoleSet:
    """RoleSet information"""
    def __init__(self, name: str, service_name: str, roleset_index: Optional[int] = None):
        self.name = name
        self.service_name = service_name
        self.roleset_index = roleset_index
        self.prefill_pods: List[PodInfo] = []
        self.decode_pods: List[PodInfo] = []

    def add_pod(self, pod: PodInfo):
        if pod.role_name == "prefill":
            self.prefill_pods.append(pod)
        elif pod.role_name == "decode":
            self.decode_pods.append(pod)
        # Technically, we also other roles rather than just P/D in scenarios like model parallisim.
        # Even in P/D, we have cls role here. To simplify the case a little bit, we will just consider P/D now.

    def remove_pod(self, pod_name: str):
        self.prefill_pods = [p for p in self.prefill_pods if p.name != pod_name]
        self.decode_pods = [p for p in self.decode_pods if p.name != pod_name]

    def get_prefill_pods(self) -> List[PodInfo]:
        return self.prefill_pods

    def get_decode_pods(self) -> List[PodInfo]:
        return self.decode_pods

class StormService:
    def __init__(self, name: str):
        self.name = name
        self.rolesets: Dict[str, RoleSet] = {}

    def add_pod(self, pod: PodInfo):
        rs = self.rolesets.setdefault(pod.roleset_name, RoleSet(pod.roleset_name, pod.service_name))
        rs.add_pod(pod)

    def remove_pod(self, pod_name: str):
        for rs in self.rolesets.values():
            rs.remove_pod(pod_name)

class PrefillConfig:
    def __init__(self, url: str, bootstrap_port: int):
        self.url = url
        self.bootstrap_port = bootstrap_port


class MiniLoadBalancer:
    def __init__(self, prefill_configs: List[PrefillConfig], decode_servers: List[str]):
        self.prefill_configs = prefill_configs
        self.prefill_servers = [p.url for p in prefill_configs]
        self.decode_servers = decode_servers

        # may support multiple stormservices
        self.stormservices: Dict[str, StormService] = {}
        self.pods: Dict[str, PodInfo] = {}
        self.models: Dict[str, StormService] = {}

    def add_pod(self, pod: PodInfo):
        self.pods[pod.name] = pod
        service = self.stormservices.setdefault(pod.service_name, StormService(pod.service_name))
        rs = service.rolesets.setdefault(
            pod.roleset_name, RoleSet(pod.roleset_name, pod.service_name, pod.roleset_index)
        )
        rs.add_pod(pod)

    def remove_pod(self, pod_name: str):
        pod = self.pods.pop(pod_name, None)
        if pod:
            svc = self.stormservices.get(pod.service_name)
            if svc:
                svc.remove_pod(pod_name)

    def get_roleset(self, roleset_name: str) -> Optional[RoleSet]:
        for svc in self.stormservices.values():
            if roleset_name in svc.rolesets:
                return svc.rolesets[roleset_name]
        return None

    def get_service_rolesets(self, service_name: str) -> List[RoleSet]:
        svc = self.stormservices.get(service_name)
        return list(svc.rolesets.values()) if svc else []

    def select_pair(self):
        prefill_config = random.choice(self.prefill_configs)
        decode_server = random.choice(self.decode_servers)
        return prefill_config.url, prefill_config.bootstrap_port, decode_server

    def select_pair_by_service(self, model_name: str):
        # TODO: find the reverse mapping, model -> service

        # find the service
        service_name = "llm-xpyd"

        svc = self.stormservices.get(service_name)
        if not svc or not svc.rolesets:
            raise HTTPException(status_code=404, detail=f"Service {service_name} can not be found")

        if len(svc.rolesets) == 1:
            # pooling mode
            rs = next(iter(svc.rolesets.values()))
            prefill = random.choice(rs.prefill_pods)
            decode = random.choice(rs.decode_pods)
        else:
            # xPyD mode
            rs = random.choice(list(svc.rolesets.values()))
            prefill = random.choice(rs.prefill_pods)
            decode = random.choice(rs.decode_pods)

        logger.info(f"model {model_name} is invoked, selected prefill {prefill.url}, decode {decode.url}")
        return prefill.url, prefill.bootstrap_port, decode.url

    async def generate(
        self, modified_request, prefill_server, decode_server, endpoint
    ) -> ORJSONResponse:
        assert endpoint[0] != "/", f"Endpoint should not start with '/': {endpoint}"

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(
                total=3600
            )  # Add timeout for request reliability
        ) as session:
            tasks = [
                session.post(f"{prefill_server}/{endpoint}", json=modified_request),
                session.post(f"{decode_server}/{endpoint}", json=modified_request),
            ]
            # Wait for both responses to complete. Prefill should end first.
            prefill_response, decode_response = await asyncio.gather(*tasks)

            return ORJSONResponse(
                content=await decode_response.json(),
                status_code=decode_response.status,
            )

    async def generate_stream(
        self, modified_request, prefill_server, decode_server, endpoint="generate"
    ):
        assert endpoint[0] != "/", f"Endpoint should not start with '/': {endpoint}"

        async def stream_results():
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(
                    total=3600
                )  # Add timeout for request reliability
            ) as session:
                try:
                    # Create the tasks for both prefill and decode requests
                    tasks = [
                        session.post(
                            f"{prefill_server}/{endpoint}", json=modified_request
                        ),
                        session.post(
                            f"{decode_server}/{endpoint}", json=modified_request
                        ),
                    ]
                    # Wait for both responses to complete. Since this is streaming, they return immediately.
                    prefill_response, decode_response = await asyncio.gather(*tasks)
                    async for chunk in decode_response.content:
                        yield chunk
                except Exception as e:
                    error_msg = {
                        "error": {"message": f"Stream processing error: {str(e)}"}
                    }
                    yield b"data: " + orjson.dumps(
                        error_msg, option=orjson.OPT_NON_STR_KEYS
                    ) + b"\n\n"
                finally:
                    if prefill_response is not None:
                        await prefill_response.release()

        return StreamingResponse(
            stream_results(),
            media_type="text/event-stream",
        )


    def watch_pods(self):
        """watch pod changes"""
        try:
            config.load_incluster_config()
        except Exception:
            config.load_kube_config()

        api = client.CoreV1Api()
        namespace = "default"
        selector = ""
        logger.info(f"Start to watch pods under namespace {namespace} + label selector {selector}.")

        watch = Watch()
        try:
            for event in watch.stream(
                api.list_namespaced_pod,
                namespace=namespace,
                label_selector=selector,
                timeout_seconds=100
            ):
                try:
                    pod = event['object']
                    pod_name = pod.metadata.name
                    event_type = event['type']

                    # filter the pods in abnormal status
                    # TODO: if pod come into deleted status, we need to remove them from the list.
                    # another apporach is to always get the full snapshot.
                    if event_type != 'DELETED' and pod.status.phase != 'Running':
                        continue

                    # extract storm service, replicaset and pod information
                    labels = pod.metadata.labels or {}
                    annotations = pod.metadata.annotations or {}

                    # TODO: unify the labels
                    role_name = labels.get('role-name', '')
                    roleset_name = labels.get('roleset-name', '')
                    service_name = labels.get('storm-service-name', '')
                    role_replica_index = annotations.get('stormservice.orchestration.aibrix.ai/role-replica-index')
                    roleset_index = annotations.get('stormservice.orchestration.aibrix.ai/roleset-index')

                    # if pod doesn't have the required annotation, skip it
                    if not all([role_name, roleset_name, service_name, role_replica_index]):
                        logger.warning(f"Pod {pod_name} doesn't have the label required, skipping")
                        continue

                    role_replica_index = int(role_replica_index)

                    # extract the default port
                    container_port = 30000
                    bootstrap_port = int(annotations.get(
                        'stormservice.orchestration.aibrix.ai/bootstrap-port', '8998'
                    ))

                    # handle pod event
                    if event_type == 'ADDED' or event_type == 'MODIFIED':
                        pod_ip = pod.status.pod_ip
                        if not pod_ip:
                            logger.warning(f"Pod {pod_name} ip is not available, skip it")
                            continue

                        pod_info = PodInfo(
                            name=pod_name,
                            ip=pod_ip,
                            port=container_port,
                            role_name=role_name,
                            roleset_name=roleset_name,
                            service_name=service_name,
                            bootstrap_port=bootstrap_port,
                            role_replica_index=role_replica_index,
                            roleset_index=int(roleset_index) if roleset_index not in (None, '') else None
                        )

                        self._update_pod(pod_info)
                        logger.info(f"Update Pod: {namespace}/{pod_name} ({event_type}), url: {pod_info.url}")

                    elif event_type == 'DELETED':
                        self._remove_pod(pod_name)
                        logger.info(f"Delete Pod: {namespace}/{pod_name}")

                except Exception as e:
                    logger.error(f"handle pod event failure: {str(e)}", exc_info=True)
        except ApiException as e:
            logger.error(f"Kubernetes API error: {e}")
        except Exception as e:
            logger.error(f"watch namespace {namespace} pod error: {str(e)}")


    def _update_pod(self, pod: PodInfo):
        """Update or add pod information in stormservices tree"""
        # remove from old roleset if exists
        if pod.name in self.pods:
            old_pod = self.pods[pod.name]
            old_svc = self.stormservices.get(old_pod.service_name)
            if old_svc:
                old_roleset = old_svc.rolesets.get(old_pod.roleset_name)
                if old_roleset:
                    if old_pod.role_name == "prefill":
                        old_roleset.prefill_pods = [p for p in old_roleset.prefill_pods if p.name != pod.name]
                    elif old_pod.role_name == "decode":
                        old_roleset.decode_pods = [p for p in old_roleset.decode_pods if p.name != pod.name]
                    # Clean up empty roleset
                    if not old_roleset.prefill_pods and not old_roleset.decode_pods:
                        del old_svc.rolesets[old_pod.roleset_name]
                # Clean up empty stormservice
                if not old_svc.rolesets:
                    del self.stormservices[old_pod.service_name]

        # Update pod dict
        self.pods[pod.name] = pod

        # Add to new roleset under stormservice
        svc = self.stormservices.setdefault(pod.service_name, StormService(pod.service_name))
        roleset = svc.rolesets.setdefault(
            pod.roleset_name, RoleSet(pod.roleset_name, pod.service_name, pod.roleset_index)
        )
        roleset.add_pod(pod)

    def _remove_pod(self, pod_name: str):
        """Remove Pod from stormservices tree"""
        if pod_name not in self.pods:
            return
        pod = self.pods[pod_name]
        svc = self.stormservices.get(pod.service_name)
        if svc:
            roleset = svc.rolesets.get(pod.roleset_name)
            if roleset:
                if pod.role_name == "prefill":
                    roleset.prefill_pods = [p for p in roleset.prefill_pods if p.name != pod_name]
                elif pod.role_name == "decode":
                    roleset.decode_pods = [p for p in roleset.decode_pods if p.name != pod_name]
                # Clean up empty roleset
                if not roleset.prefill_pods and not roleset.decode_pods:
                    del svc.rolesets[pod.roleset_name]
            # Clean up empty stormservice
            if not svc.rolesets:
                del self.stormservices[pod.service_name]

        del self.pods[pod_name]


app = FastAPI()
load_balancer = None

# @app.on_event("startup")
# async def start_watchers():
#     logger.info("Startup hook triggered, starting to watch pods...")
#     asyncio.create_task(load_balancer.watch_pods())


@app.get("/health")
async def health_check():
    return Response(status_code=200)


@app.get("/health_generate")
async def health_check():
    prefill_servers, decode_servers = (
        load_balancer.prefill_servers,
        load_balancer.decode_servers,
    )
    async with aiohttp.ClientSession() as session:
        # Create the tasks
        tasks = []
        for server in chain(prefill_servers, decode_servers):
            tasks.append(session.post(f"{server}/health_generate"))
        for i, response in enumerate(asyncio.as_completed(tasks)):
            await response
    return Response(status_code=200)


@app.post("/flush_cache")
async def flush_cache():
    prefill_servers, decode_servers = (
        load_balancer.prefill_servers,
        load_balancer.decode_servers,
    )
    async with aiohttp.ClientSession() as session:
        # Create the tasks
        tasks = []
        for server in chain(prefill_servers, decode_servers):
            tasks.append(session.post(f"{server}/flush_cache"))
        for i, response in enumerate(asyncio.as_completed(tasks)):
            await response
    return Response(status_code=200)


@app.get("/servers")
async def get_servers():
    stormservice_view = {}
    for svc_name, svc in load_balancer.stormservices.items():
        roleset_view = {}
        for rs_name, rs in svc.rolesets.items():
            prefill_pods = [
                {
                    "name": pod.name,
                    "ip": pod.ip,
                    "port": pod.port,
                    "bootstrap_port": pod.bootstrap_port,
                    "role_replica_index": pod.role_replica_index,
                    "roleset_index": pod.roleset_index,
                }
                for pod in rs.prefill_pods
            ]
            decode_pods = [
                {
                    "name": pod.name,
                    "ip": pod.ip,
                    "port": pod.port,
                    "bootstrap_port": pod.bootstrap_port,
                    "role_replica_index": pod.role_replica_index,
                    "roleset_index": pod.roleset_index,
                }
                for pod in rs.decode_pods
            ]
            roleset_view[rs_name] = {
                "prefill_pods": prefill_pods,
                "decode_pods": decode_pods
            }

        stormservice_view[svc_name] = {
            "rolesets": roleset_view
        }

    return ORJSONResponse(
        content={
            "stormservices": stormservice_view
        }
    )

@app.get("/get_server_info")
async def get_server_info():
    prefill_servers, decode_servers = (
        load_balancer.prefill_servers,
        load_balancer.decode_servers,
    )
    prefill_infos = []
    decode_infos = []
    async with aiohttp.ClientSession() as session:
        for server in chain(prefill_servers):
            server_info = await session.get(f"{server}/get_server_info")
            prefill_infos.append(await server_info.json())
        for server in chain(decode_servers):
            server_info = await session.get(f"{server}/get_server_info")
            decode_infos.append(await server_info.json())

    return {"prefill": prefill_infos, "decode": decode_infos}


@app.get("/get_model_info")
async def get_model_info():
    # Dummy model information
    model_info = {
        "model_path": "/path/to/dummy/model",
        "tokenizer_path": "/path/to/dummy/tokenizer",
        "is_generation": True,
        "preferred_sampling_params": {"temperature": 0.7, "max_new_tokens": 128},
    }
    return ORJSONResponse(content=model_info)


@app.post("/generate")
async def handle_generate_request(request_data: dict):
    # prefill_server, bootstrap_port, decode_server = load_balancer.select_pair()
    logger.info("GET request {request_data}")
    model_name = request_data['model']
    prefill_server, bootstrap_port, decode_server = load_balancer.select_pair_by_service(model_name)

    # Parse and transform prefill_server for bootstrap data
    parsed_url = urllib.parse.urlparse(prefill_server)
    hostname = parsed_url.hostname
    modified_request = request_data.copy()

    batch_size = _get_request_batch_size(modified_request)
    if batch_size is not None:
        modified_request.update(
            {
                "bootstrap_host": [hostname] * batch_size,
                "bootstrap_port": [bootstrap_port] * batch_size,
                "bootstrap_room": [
                    _generate_bootstrap_room() for _ in range(batch_size)
                ],
            }
        )
    else:
        modified_request.update(
            {
                "bootstrap_host": hostname,
                "bootstrap_port": bootstrap_port,
                "bootstrap_room": _generate_bootstrap_room(),
            }
        )

    if request_data.get("stream", False):
        return await load_balancer.generate_stream(
            modified_request, prefill_server, decode_server, "generate"
        )
    else:
        return await load_balancer.generate(
            modified_request, prefill_server, decode_server, "generate"
        )


@app.post("/v1/chat/completions")
async def handle_completion_request(request_data: dict):
    # prefill_server, bootstrap_port, decode_server = load_balancer.select_pair()
    model_name = request_data['model']
    logger.info(f"GET request {request_data}, requested model {model_name}")
    prefill_server, bootstrap_port, decode_server = load_balancer.select_pair_by_service(model_name)

    # Parse and transform prefill_server for bootstrap data
    parsed_url = urllib.parse.urlparse(prefill_server)
    hostname = parsed_url.hostname
    modified_request = request_data.copy()
    modified_request.update(
        {
            "bootstrap_host": hostname,
            "bootstrap_port": bootstrap_port,
            "bootstrap_room": random.randint(0, 2**63 - 1),
        }
    )

    if request_data.get("stream", False):
        return await load_balancer.generate_stream(
            modified_request,
            prefill_server,
            decode_server,
            endpoint="v1/chat/completions",
        )
    else:
        return await load_balancer.generate(
            modified_request,
            prefill_server,
            decode_server,
            endpoint="v1/chat/completions",
        )


def _generate_bootstrap_room():
    return random.randint(0, 2**63 - 1)


# We may utilize `GenerateReqInput`'s logic later
def _get_request_batch_size(request):
    if (text := request.get("text")) is not None:
        return None if isinstance(text, str) else len(text)
    if (input_ids := request.get("input_ids")) is not None:
        return None if isinstance(input_ids[0], int) else len(input_ids)
    return None


@app.get("/v1/models")
async def get_models():
    prefill_server = load_balancer.prefill_servers[0]  # Get the first prefill server
    async with aiohttp.ClientSession() as session:
        try:
            response = await session.get(f"{prefill_server}/v1/models")
            if response.status != 200:
                raise HTTPException(
                    status_code=response.status,
                    detail=f"Prefill server error: Status {response.status}",
                )
            return ORJSONResponse(content=await response.json())
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


def run(prefill_configs, decode_addrs, host, port):
    global load_balancer
    load_balancer = MiniLoadBalancer(prefill_configs, decode_addrs)
    logger.info("Startup hook triggered, starting to watch pods...")
    watch_thread = Thread(target=load_balancer.watch_pods)
    watch_thread.start()
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Mini Load Balancer Server")
    parser.add_argument(
        "--prefill", required=False, help="Comma-separated URLs for prefill servers"
    )
    parser.add_argument(
        "--prefill-bootstrap-ports",
        help="Comma-separated bootstrap ports for prefill servers",
        default="8998",
    )
    parser.add_argument(
        "--decode", required=False, help="Comma-separated URLs for decode servers"
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host to bind the server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to bind the server (default: 8000)"
    )
    args = parser.parse_args()
    prefill_configs = []
    decode_addrs = []
    if args.prefill and args.decode:
        prefill_urls = args.prefill.split(",")
        bootstrap_ports = [int(p) for p in args.prefill_bootstrap_ports.split(",")]

        if len(bootstrap_ports) == 1:
            bootstrap_ports = bootstrap_ports * len(prefill_urls)
        else:
            if len(bootstrap_ports) != len(prefill_urls):
                raise ValueError(
                    "Number of prefill URLs must match number of bootstrap ports"
                )
                exit(1)

        for url, port in zip(prefill_urls, bootstrap_ports):
            prefill_configs.append(PrefillConfig(url, port))

        decode_addrs = args.decode.split(",")

    run(prefill_configs, decode_addrs, args.host, args.port)
