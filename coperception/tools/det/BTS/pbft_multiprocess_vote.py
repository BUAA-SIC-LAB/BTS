# -*- coding: utf-8 -*-
"""
Time        :   2025/03/11 13:21
Author      :   lzh594(£·)
Version     :
File        :   pbft_multiprocess_vote.py
Describe    :
"""
import json
import logging
import os
import select
from logging.handlers import QueueListener, QueueHandler
from typing import Union

import numpy as np
import torch
from torch import multiprocessing as mp
import socket
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum

from tools.det.BTS.BTS_util import check_folder, time_str, init_model, sus_check_jaccard

# f
TOLERANT = 1
# n
NUM_NODES = 3 * TOLERANT + 1
# CLIENT ID
CLIENT_ID = -1
# large buf
BUFFER_SIZE = 4192


class NodeState(Enum):
    """Node states in PBFT protocol"""
    REQUEST = "REQUEST"
    INIT = "INIT"
    PRE_PREPARE = "PRE_PREPARE"
    PREPARE = "PREPARE"
    COMMIT = "COMMIT"
    REPLY = "REPLY"


@dataclass
class Message:
    """Message structure for PBFT communication"""
    node_id: int
    stage: str
    content: Union[str, dict]
    sequence_number: int
    timestamp: float


class PBFTNode:
    """
    PBFT Node structure
    """

    def __init__(self, node_id, port, peers, client_port, tolerant, byzantine_ids, current_sequence, is_alive, is_ready, log_queue, state, shared_data, config, args, primary):
        self.node_id = node_id
        self.port = port
        self.peers = peers
        self.client_port = client_port
        self.tolerant = tolerant
        self.primary = primary  # Primary node ID
        self.state = state
        self.is_alive = is_alive
        self.is_ready = is_ready
        self.byzantine_ids = byzantine_ids
        self.current_sequence = current_sequence
        self.prepared_messages = dict()  # Track prepared messages by sequence
        self.committed_messages = dict()  # Track committed messages by sequence
        self.suspected_nodes_votes = dict()
        self.server_socket = None
        self.shared_data = shared_data
        self.device =  torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.args = args
        num_agent = len(peers)
        _, fafmodule = init_model(self.config, self.args, num_agent, self.device) if config is not None else (None, None)
        self.model = fafmodule
        self.logger = self.set_log(log_queue, f'Node-{self.node_id}' if node_id != CLIENT_ID else "Client" )
        # debug
        self.check_debug_times = []
        # print(f"CUDA_VISIBLE_DEVICES in Node {self.node_id}: {os.environ.get('CUDA_VISIBLE_DEVICES')}")


    @staticmethod
    def set_log(log_queue, name):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        queue_handler = QueueHandler(log_queue)
        logger.addHandler(queue_handler)
        return logger

    @staticmethod
    def data_to_device(data_no_device, device):
        data = dict()
        keys_to_clone = ["bev_seq", "bev_seq_teacher", "labels", "reg_targets", "anchors", "vis_maps", "reg_loss_mask", "target_agent_ids", "num_agent", "trans_matrices", "pert"]
        keys_to_copy = ["ego_agent", "no_fuse", "collab_agent_list", "trial_agent_id", "confidence", "unadv_pert", "attacker_list", "eps"]
        for key in keys_to_clone:
            if data_no_device[key] is not None:
                data[key] = data_no_device[key].clone().to(device)
            else:
                data[key] = None
        for key in keys_to_copy:
            data[key] = data_no_device[key]
        return data

    def prepare_check(self):
        # print(f"Node {self.node_id}: device: {self.device}")
        # print(f"CUDA_VISIBLE_DEVICES in Node {self.node_id}: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
        data_no_device = self.shared_data["data_no_device"]
        data = self.data_to_device(data_no_device, self.device)
        other_data = self.shared_data["other_data"]
        jac_data = self.shared_data.get("jac_data")
        if jac_data is not None:
            return data, other_data["num_agent"], self.tolerant, other_data["root_logger_dilled"], jac_data, self.args.box_matching_thresh
        return data, other_data["num_agent"], other_data["root_logger_dilled"]

    def start_server_listening(self):
        """Start server to listen for incoming connections"""
        """
            AF_INET: Indicates the use of IPv4 addresses (replace with AF_INET6 if using IPv6)  
            SOCK_STREAM: Indicates the use of TCP (connection-oriented stream communication)  
             O_REUSEADDR: It allows quick reuse of ports, preventing "Address already in use" errors
        """
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(('localhost', self.port))
        server_socket.listen()
        self.server_socket = server_socket
        if self.is_ready is not None:
            self.is_ready.set()
        self.logger.info(f"is listening on port {self.port}")
        while not self.is_alive.is_set():
            try:
                readable, _, _ = select.select([server_socket], [], [], 0.1)
                if readable:
                    conn, addr = server_socket.accept()
                    t = threading.Thread(target=self.handle_client, args=(conn,), name="handle_client")
                    t.start()
                    t.join()
            except Exception as e:
                self.logger.error(f"Error accepting connection: {e}")
                self.logger.error(traceback.format_exc())
        else:
            self.logger.info(f"is shutting down")
            self.server_socket.close()


    def handle_client(self, conn: socket.socket):
        """Handle incoming client connection"""
        try:
            data = ""
            while True:
                chunk = conn.recv(BUFFER_SIZE).decode('utf-8')
                data += chunk
                if len(chunk) < BUFFER_SIZE:
                    break
            msg_obj = Message(**(json.loads(data)))
            # attacked agent(s) don't process
            if self.node_id not in self.byzantine_ids:
                self.process_message(msg_obj)
        except Exception as e:
            self.logger.error(f"Error handling client: {e}")
            self.logger.error(traceback.format_exc())
        finally:
            conn.close()

    def send_message(self, peer_port: int, message: Message):
        """Send message to specific peer"""
        try:
            send_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            send_socket.settimeout(5)  # 5 second timeout
            send_socket.connect(('localhost', peer_port))
            send_socket.send(json.dumps(message.__dict__).encode('utf-8'))
            send_socket.close()
        except Exception as e:
            self.logger.error(f"Error sending message to port {peer_port}: {e}")
            self.logger.error(traceback.format_exc())

    def broadcast_message(self, stage: NodeState, content: str):
        """Broadcast message to all peers"""
        message = Message(node_id=self.node_id, stage=stage.value, content=content, sequence_number=self.current_sequence.value, timestamp=time.time())
        peers_without_self = [(peer_port, message) for peer_id, peer_port in self.peers.items() if peer_id not in [self.node_id, CLIENT_ID]]
        with ThreadPoolExecutor(max_workers=len(peers_without_self)) as executor:
            futures = [executor.submit(self.send_message,peer_port, message) for peer_port, message in peers_without_self]
            for future in futures:
                future.result()
        self.logger.info(f"broadcast {message.sequence_number}th {message.stage} message to peers: {[peer_id for peer_id in self.peers.keys() if peer_id not in [self.node_id, CLIENT_ID]]}")

    def process_message(self, message):
        """Process received message according to PBFT protocol"""
        if message.sequence_number < self.current_sequence.value:
            self.logger.warning("Outdated message received, ignoring")
            return
        if message.stage == NodeState.REQUEST.value and self.state.state == NodeState.INIT:
            self.handle_request(message)
        elif message.stage == NodeState.PRE_PREPARE.value and self.state.state == NodeState.INIT:
            self.handle_pre_prepare(message)
        elif message.stage == NodeState.PREPARE.value and self.state.state == NodeState.PRE_PREPARE:
            self.handle_prepare(message)
        elif message.stage == NodeState.COMMIT.value and self.state.state == NodeState.PREPARE:
            self.handle_commit(message)
        else:
            # FIXME:
            # self.logger.info(f'{message.sequence_number}th msg: unknown stage {message.stage} or state {self.state} wrong from node {message.node_id}')
            return
        self.logger.info(f"received {message.sequence_number}th {message.stage} from node {message.node_id}")

    def handle_request(self, message: Message):
        self.broadcast_message(NodeState.PRE_PREPARE, message.content)
        if self.shared_data.get("data_no_device") is not None:
            st = time.time()
            check_data = self.prepare_check()
            sus_check = sus_check_jaccard(self.config, self.model, check_data, self.check_debug_times)
            self.logger.info(f"suspicion check in: {time.time() - st:.4f}")
            self.state.state = NodeState.PRE_PREPARE
            votes_seq_list = self.suspected_nodes_votes.setdefault(message.sequence_number, list())
            votes_seq_list.extend(sus_check)
        else:
            self.state.state = NodeState.PRE_PREPARE
            votes_seq_list = self.suspected_nodes_votes.setdefault(message.sequence_number, list())
            votes_seq_list.extend(self.byzantine_ids)

    def handle_pre_prepare(self, message: Message):
        """Handle pre-prepare phase"""
        if self.shared_data.get("data_no_device") is not None:
            st = time.time()
            check_data = self.prepare_check()
            sus_check = sus_check_jaccard(self.config, self.model, check_data, self.check_debug_times)
            self.logger.info(f"suspicion check in: {time.time() - st:.4f}")
            votes_seq_list = self.suspected_nodes_votes.setdefault(message.sequence_number, list())
            votes_seq_list.extend(sus_check)
            self.state.state = NodeState.PRE_PREPARE
            self.prepared_messages.setdefault(message.sequence_number, list()).append(message.node_id)
            self.broadcast_message(NodeState.PREPARE, message.content)
            self.logger.info(f"think {sus_check} was attacked")
        else:
            votes_seq_list = self.suspected_nodes_votes.setdefault(message.sequence_number, list())
            votes_seq_list.extend(self.byzantine_ids)
            self.state.state = NodeState.PRE_PREPARE
            self.prepared_messages.setdefault(message.sequence_number, list()).append(message.node_id)
            self.broadcast_message(NodeState.PREPARE, message.content)

    def handle_prepare(self, message: Message):
        """Handle prepare phase"""
        self.prepared_messages.setdefault(message.sequence_number, list()).append(message.node_id)
        prepared_count = len(self.prepared_messages[message.sequence_number])
        if prepared_count >= 2 * self.tolerant:
            self.logger.info(f"have received {prepared_count} prepared messages")
            self.state.state = NodeState.PREPARE
            self.broadcast_message(NodeState.COMMIT, message.content)

    def handle_commit(self, message: Message):
        """Handle commit phase"""
        self.committed_messages.setdefault(message.sequence_number, list()).append(message.node_id)
        committed_count = len(self.committed_messages[message.sequence_number])
        if committed_count >= 2 * self.tolerant:
            self.state.state = NodeState.COMMIT
            self.reply(self.suspected_nodes_votes[message.sequence_number])

    def reply(self, content):
        """Make final decision"""
        self.logger.info(f"make {self.current_sequence.value}th consensus, suspect {content}")
        # Notify client
        client_message = Message(node_id=self.node_id, stage=NodeState.REPLY.value, content=content, sequence_number=self.current_sequence.value, timestamp=time.time())
        self.send_message(self.client_port, client_message)
        self.state.state = NodeState.REPLY

    def run(self):
        t = threading.Thread(target=self.start_server_listening, daemon=True, name=f"{self.node_id} server")
        t.start()
        t.join()
        self.logger.info(f"check predict avg time: {np.mean(self.check_debug_times) if len(self.check_debug_times) != 0 else 0: 4f}")


class PBFTClient(PBFTNode):
    """Client node for PBFT network"""

    def __init__(self, client_id, client_port, peers, tolerant, byzantine_ids, current_sequence, client_items, is_alive, log_queue, primary):
        super().__init__(client_id, client_port, peers, peers[CLIENT_ID], tolerant, byzantine_ids, current_sequence, is_alive, None, log_queue, NodeState.INIT, None, None, None, primary)
        self.response_received = client_items["client_event"]
        self.condition = client_items["client_condition"]
        self.consensus = client_items["client_consensus"]  # Track prepared messages by sequence
        self.top_f_sus = []

    def send_request(self, request):
        """Send request to primary node"""
        message= Message(node_id=self.node_id, stage=NodeState.REQUEST.value, content=request,
                   sequence_number=self.current_sequence.value, timestamp=time.time())
        self.send_message(self.peers[self.primary], message)
        self.logger.info(f"sent request: {request}")

    def process_message(self, message):
        """Handle response from PBFT network"""
        with self.condition:
            if message.stage == NodeState.REPLY.value:
                self.consensus.setdefault(message.sequence_number, list()).append(message.node_id)
                consensus_count = len(self.consensus[message.sequence_number])
                votes_seq_dict = self.suspected_nodes_votes.setdefault(message.sequence_number, dict())
                for sus_id in message.content:
                    votes_seq_dict[sus_id] = votes_seq_dict.get(sus_id, 0) + 1
                self.logger.info(f"received consensus: {message.sequence_number}th msg {message.content} from node {message.node_id}")
                # TODO: Wait as much as possible for the results
                if consensus_count >= self.tolerant+1 and not self.response_received.is_set():
                    self.response_received.set()
                    self.condition.notify()
                    self.logger.info(f"have made {message.sequence_number}th {consensus_count} consensus messages")
                    self.logger.info(f"count votes: {votes_seq_dict}")


class PBFTProtocol:
    """PBFT protocol"""
    def __init__(self, config, args, num_agent, tolerant_nodes, all_agent_list, byzantine_ids, primary):
        self.config = config
        self.args = args
        self.total_nodes = num_agent-1
        self.tolerant_nodes = tolerant_nodes
        self.nodes = list(all_agent_list) # avoid the modification outside
        self.primary = primary
        self.client_id = CLIENT_ID
        self.client = None
        self.manager = mp.Manager()
        self.byzantine_ids =  self.manager.list(byzantine_ids)
        self.running = self.manager.Event()
        self.current_sequence = self.manager.Value("i", 0)
        self.shared_data = self.manager.dict()
        self.states = []
        self.share = {}
        self.processes = []
        self.avg_times = []
        self.logger = None
        self.queue_listener = None

    def init(self):
        """Initialize PBFT network"""
        # initialize logger
        log_queue, self.queue_listener = self.setup_logging()
        self.logger.info(f"set `BYZANTINE_IDS`: {self.byzantine_ids}")
        # initialize params
        # self.nodes: [..., -1] -1是client
        self.nodes.append(self.client_id)
        peers = {node_id: 55000 + node_id * 10 for node_id in self.nodes}
        print(f"peers: {peers}")
        for _ in range(self.total_nodes + 1):
            state_named = self.manager.Namespace()
            state_named.state = NodeState.INIT
            self.states.append(state_named)
        # Initialize Client
        self.share["client_share"] = {
            "client_items": self.manager.dict({
                "client_event": self.manager.Event(),
                "client_condition": self.manager.Condition(),
                "client_consensus": dict(), }),
            "client_id": self.client_id, "client_port": peers[self.client_id], "primary": self.primary,
            "peers": peers, "tolerant": self.tolerant_nodes, "byzantine_ids": self.byzantine_ids,
            "current_sequence": self.current_sequence,"is_alive": self.running, "log_queue": log_queue,
        }
        self.client = PBFTClient(**(self.share["client_share"]))
        threading.Thread(target=self.client.start_server_listening, name="client_server",daemon=True).start()
        # Initialize Nodes
        self.share["nodes_share"] = [
            {"node_id": node_id, "port": peers[node_id], "peers": peers,
             "client_port": peers[self.client_id], "tolerant": self.tolerant_nodes,
             "byzantine_ids":self.byzantine_ids, "is_alive": self.running, "primary": self.primary,
             "current_sequence": self.current_sequence, "is_ready": self.manager.Event(),
             "log_queue": log_queue, "state": self.states[node_id], "config": self.config,
             "shared_data": self.shared_data, "args": self.args}
            for node_id in self.nodes[:self.total_nodes]]
        # Start processes: CUDA must be in spawn not fork
        context = mp.get_context("spawn")
        for id_idx, node_id in enumerate(self.nodes[:self.total_nodes]):
            # target cannot contain self (weak reference), set the child process environment variables before starting, GPU 1, 2, 3 are used by the child process
            os.environ["CUDA_VISIBLE_DEVICES"] = str(id_idx % 3+1)
            ctx = context.Process(target=self.start_node_process, args=(self.share["nodes_share"][id_idx],), name=f"{node_id}")
            ctx.start()
            self.processes.append(ctx)

    @staticmethod
    def start_node_process(node_param):
        node = PBFTNode(**node_param)
        node.run()

    def update_pbft(self, byzantine_ids, data_no_device, other_data, jac_data=None):
        """ update the param byzantine_ids"""
        self.client.response_received.clear()
        del self.byzantine_ids[:]
        self.byzantine_ids.extend(byzantine_ids)
        self.logger.info(f"update BYZANTINE_IDS: {byzantine_ids}")
        if data_no_device is not None:
            self.shared_data["data_no_device"] = self.manager.dict(data_no_device)
            self.shared_data["other_data"] = self.manager.dict(other_data)
        if jac_data is not None:
            self.shared_data["jac_data"] = self.manager.dict(jac_data)
        self.current_sequence.value += 1
        for node_state in self.states:
            node_state.state = NodeState.INIT

    def run(self, req_msg):
        """Initialize and run PBFT network"""
        st = time.time()
        # wait the server ready
        while not all(event.is_set() for event in [d.get("is_ready") for d in self.share["nodes_share"]]):
            time.sleep(0.001)
        self.client.send_request(req_msg)
        # wait the reply flag
        with self.client.condition:
            self.client.condition.wait_for(lambda: self.client.response_received.is_set())
        res_consensus = self.client.consensus.get(self.client.current_sequence.value)
        votes_seq_dict = self.client.suspected_nodes_votes.get(self.client.current_sequence.value)
        top_f_sus = []
        if len(votes_seq_dict) > self.tolerant_nodes:
            self.client.logger.info(f"The number of suspected agents({votes_seq_dict.keys()}) exceeds f: {len(votes_seq_dict)}>{self.tolerant_nodes})")
        else:
            self.client.logger.info(f"make consensus(es)! with Agents idx {res_consensus}!")
            top_f_sus = set([idx for idx, vote in votes_seq_dict.items()])
            self.client.logger.info(f"suspected nodes(top f:{self.tolerant_nodes}): {top_f_sus}")
        et = time.time()
        self.avg_times.append(et-st)
        self.logger.info(f"protocol run in {et - st:.4f} seconds")
        time.sleep(0.1)
        return res_consensus, top_f_sus


    def clean(self):
        self.running.set()
        for p in self.processes:
            p.join()
        self.queue_listener.stop()

    def setup_logging(self):
        log_queue = self.manager.Queue()
        self.logger = logging.getLogger("PBFT")
        self.logger.setLevel(logging.INFO)
        queue_handler = QueueHandler(log_queue)
        self.logger.addHandler(queue_handler)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s %(message)s')
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        log_dir = '/home/liuzhenghao/BTS/coperception/tools/det/BTS/pbft_log'
        check_folder(log_dir)
        filename=os.path.join(log_dir, f"pbft_test_f_{TOLERANT}_n_{NUM_NODES}_{time_str()}.log")
        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(formatter)
        queue_listener = QueueListener(log_queue, stream_handler, file_handler)
        queue_listener.start()

        return log_queue, queue_listener


if __name__ == '__main__':
    assert NUM_NODES >= 3 * TOLERANT + 1
    BYZANTINE_IDS = [0]
    nodes = [0,1,3,4,5]
    primary = 1
    pbft = PBFTProtocol(None, None, len(nodes), TOLERANT, nodes, BYZANTINE_IDS, primary)
    pbft.init()
    for i in range(10):
        request_msg = f"this is {i}th request message"
        pbft.run(request_msg)
        # time.sleep(0.1)
        print(f"update: {i}th")
        pbft.update_pbft([i%3+3],None,None)
    pbft.clean()