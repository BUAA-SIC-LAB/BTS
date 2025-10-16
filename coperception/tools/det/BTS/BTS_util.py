import logging
import random
import argparse
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
coperception_dir = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
sys.path.append(coperception_dir)

import dill
import torch.optim as optim
from torch.utils.data import DataLoader

from coperception.configs import Config, ConfigGlobal
from coperception.datasets import V2XSimDet
from coperception.models.det import *
from coperception.utils.loss import *
from coperception.utils.CoDetModule import *
from coperception.utils.mean_ap import eval_map
from tools.det.box_matching import associate_2_detections


def check_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
    return folder_path


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # global setting！！！
    torch.backends.cudnn.deterministic = True


def get_jaccard_index(config, ego_agent, num_agent_list, padded_voxel_point, reg_target, anchors_map, gt_max_iou, result_1, result_2):
    num_sensor = num_agent_list[0][0].numpy()
    det_results_local_1 = [[] for _ in range(num_sensor)]
    annotations_local_1 = [[] for _ in range(num_sensor)]
    det_results_local_2 = [[] for _ in range(num_sensor)]
    annotations_local_2 = [[] for _ in range(num_sensor)]
    ego_idx = ego_agent
    # for k in range(num_sensor):
    data_agents = {'bev_seq': torch.unsqueeze(padded_voxel_point[ego_idx, :, :, :, :], 1), 'reg_targets': torch.unsqueeze(reg_target[ego_idx, :, :, :, :, :], 0),
                   'anchors': torch.unsqueeze(anchors_map[ego_idx, :, :, :, :], 0)}
    temp = gt_max_iou[ego_idx]
    data_agents['gt_max_iou'] = temp[0]['gt_box'][0, :, :]
    result_temp_1 = result_1[ego_idx]
    result_temp_2 = result_2[ego_idx]
    temp_1 = {'bev_seq': data_agents['bev_seq'][0, -1].cpu().numpy(), 'result': result_temp_1[0][0], 'reg_targets': data_agents['reg_targets'].cpu().numpy()[0],
              'anchors_map': data_agents['anchors'].cpu().numpy()[0], 'gt_max_iou': data_agents['gt_max_iou']}
    temp_2 = {'bev_seq': data_agents['bev_seq'][0, -1].cpu().numpy(), 'result': result_temp_2[0][0], 'reg_targets': data_agents['reg_targets'].cpu().numpy()[0],
              'anchors_map': data_agents['anchors'].cpu().numpy()[0], 'gt_max_iou': data_agents['gt_max_iou']}

    det_results_local_1[ego_idx], annotations_local_1[ego_idx] = cal_local_mAP(config, temp_1, det_results_local_1[ego_idx], annotations_local_1[ego_idx])
    det_results_local_2[ego_idx], annotations_local_2[ego_idx] = cal_local_mAP(config, temp_2, det_results_local_2[ego_idx], annotations_local_2[ego_idx])

    print("Calculating in the view of Agent {}:".format(ego_idx))
    # shape of det_results_local_1 [k][0][0] is (N, 9)
    # The final value of the array is confidence. Ignored
    if len(det_results_local_1[ego_idx]) == 0:
        # if ego have no detection, return 0
        return 0
    det_1 = det_results_local_1[ego_idx][0][0][:, 0:8]
    det_2 = det_results_local_2[ego_idx][0][0][:, 0:8]
    # jac_index = calculate_jaccard(det_results_local_1[k][0][0], det_results_local_2[k][0][0])
    jac_index = associate_2_detections(det_1, det_2)
    return jac_index

def get_jaccard_only(config, ego_idx, num_agent_list, padded_voxel_point, reg_target, anchors_map, gt_max_iou, result):
    # 获取代理数量
    num_sensor = len(num_agent_list)
    det_results_local_1 = [[] for _ in range(num_sensor)]
    annotations_local_1 = [[] for _ in range(num_sensor)]
    det_results_local_2 = [[] for _ in range(num_sensor)]
    annotations_local_2 = [[] for _ in range(num_sensor)]

    # 处理 ego 代理的数据
    data_agents = {'bev_seq': torch.unsqueeze(padded_voxel_point[ego_idx, :, :, :, :], 1),
        'reg_targets': torch.unsqueeze(reg_target[ego_idx, :, :, :, :, :], 0),
        'anchors': torch.unsqueeze(anchors_map[ego_idx, :, :, :, :], 0)}
    temp = gt_max_iou[ego_idx]
    data_agents['gt_max_iou'] = temp[0]['gt_box'][0, :, :]
    result_temp_1 = result[ego_idx]
    temp_1 = {
        'bev_seq': data_agents['bev_seq'][0, -1].cpu().numpy(),
        'result': result_temp_1[0][0],
        'reg_targets': data_agents['reg_targets'].cpu().numpy()[0],
        'anchors_map': data_agents['anchors'].cpu().numpy()[0],
        'gt_max_iou': data_agents['gt_max_iou']
    }
    det_results_local_1[ego_idx], annotations_local_1[ego_idx] = cal_local_mAP(
        config, temp_1, det_results_local_1[ego_idx], annotations_local_1[ego_idx]
    )
    # If the ego agent has no detection results, return a list of all 0s
    if len(det_results_local_1[ego_idx]) == 0:
        jac_indices = [0.0 for _ in range(num_sensor - 1)]
        jac_indices.insert(ego_idx, 1.0)
        return jac_indices
    # Extract the detection box information of the ego agent
    det_1 = det_results_local_1[ego_idx][0][0][:, 0:8]

    jac_indices = []
    # Traverse other proxies except ego proxy
    for other_agent in range(num_sensor):
        data_other_agents = {
            'bev_seq': torch.unsqueeze(padded_voxel_point[other_agent, :, :, :, :], 1),
            'reg_targets': torch.unsqueeze(reg_target[other_agent, :, :, :, :, :], 0),
            'anchors': torch.unsqueeze(anchors_map[other_agent, :, :, :, :], 0)
        }
        temp_other = gt_max_iou[other_agent]
        data_other_agents['gt_max_iou'] = temp_other[0]['gt_box'][0, :, :]
        result_temp_2 = result[other_agent]
        temp_2 = {
            'bev_seq': data_other_agents['bev_seq'][0, -1].cpu().numpy(),
            'result': result_temp_2[0][0],
            'reg_targets': data_other_agents['reg_targets'].cpu().numpy()[0],
            'anchors_map': data_other_agents['anchors'].cpu().numpy()[0],
            'gt_max_iou': data_other_agents['gt_max_iou']
        }
        det_results_local_2[other_agent], annotations_local_2[other_agent] = cal_local_mAP(
            config, temp_2, det_results_local_2[other_agent], annotations_local_2[other_agent]
        )
        # If there is no detection result from other agents, the similarity is recorded as 0
        if len(det_results_local_2[other_agent]) == 0:
            jac_indices.append(0.0)
            continue
        # Extract the detection box information of other current agents
        det_2 = det_results_local_2[other_agent][0][0][:, 0:8]
        # Calculate Jaccard index
        jac_index = associate_2_detections(det_1, det_2)
        jac_indices.append(jac_index)

    return jac_indices


def visualize(config, args_visualization, filename0, save_fig_path, fafmodule, data, num_agent_list, gt_max_iou, vis_tag):
    print("Visualizing: {}".format(vis_tag))
    num_sensor = len(num_agent_list)

    det_results_local = [[] for _ in range(num_sensor)]
    annotations_local = [[] for _ in range(num_sensor)]

    padded_voxel_point = data['bev_seq']
    padded_voxel_points_teacher = data['bev_seq_teacher']
    reg_target = data['reg_targets']
    anchors_map = data['anchors']

    loss, cls_loss, loc_loss, result = fafmodule.predict_all(data, 1, num_agent=len(num_agent_list))

    # local qualitative evaluation
    print(f'num_sensor: {num_sensor}')
    for k in range(num_sensor):
        # debug: just output ego
        if k != data["ego_agent"]:
            continue
        data_agents = {'bev_seq': torch.unsqueeze(padded_voxel_point[k, :, :, :, :], 1), 'bev_seq_teacher': torch.unsqueeze(padded_voxel_points_teacher[k, :, :, :, :], 1),
                       'reg_targets': torch.unsqueeze(reg_target[k, :, :, :, :, :], 0), 'anchors': torch.unsqueeze(anchors_map[k, :, :, :, :], 0)}
        temp = gt_max_iou[k]
        data_agents['gt_max_iou'] = temp[0]['gt_box'][0, :, :]
        result_temp = result[k]

        temp = {'bev_seq': data_agents['bev_seq'][0, -1].cpu().numpy(), 'bev_seq_teacher': data_agents['bev_seq_teacher'][0, -1].cpu().numpy(), 'result': result_temp[0][0],
                'reg_targets': data_agents['reg_targets'].cpu().numpy()[0], 'anchors_map': data_agents['anchors'].cpu().numpy()[0], 'gt_max_iou': data_agents['gt_max_iou'], 'vis_tag': vis_tag}

        det_results_local[k], annotations_local[k] = cal_local_mAP(config, temp, det_results_local[k], annotations_local[k])
        print("Agent {}:".format(k))
        filename = str(filename0[0][0])
        cut = filename[filename.rfind('agent') + 7:]
        seq_name = cut[:cut.rfind('_')]
        idx = cut[cut.rfind('_') + 1:cut.rfind('/')]
        seq_save = os.path.join(save_fig_path[k], seq_name)
        check_folder(seq_save)
        idx_save = '{}_{}.png'.format(str(idx), vis_tag)

        if args_visualization:
            visualization(config, temp, None, None, 0, os.path.join(seq_save, idx_save))


def time_str():
    # t = time.time() - 60 * 60 * 24 * 30
    t = time.time()
    time_string = time.strftime("%Y%m%d_%H:%M:%S", time.localtime(t))
    return time_string


def local_eval(num_agent, padded_voxel_points, reg_target, anchors_map, gt_max_iou, result, config, det_results_local, annotations_local):
    # If has RSU, do not count RSU's output into evaluation
    # eval_start_idx = 0 if args.no_cross_road else 1
    eval_start_idx = 0
    # update global result
    for k in range(eval_start_idx, num_agent):
        data_agents = {
            "bev_seq": torch.unsqueeze(padded_voxel_points[k, :, :, :, :], 1),
            "reg_targets": torch.unsqueeze(reg_target[k, :, :, :, :, :], 0),
            "anchors": torch.unsqueeze(anchors_map[k, :, :, :, :], 0),
        }
        temp = gt_max_iou[k]

        if len(temp[0]["gt_box"]) == 0:
            data_agents["gt_max_iou"] = []
        else:
            data_agents["gt_max_iou"] = temp[0]["gt_box"][0, :, :]


        result_temp = result[k]

        temp = {
            "bev_seq": data_agents["bev_seq"][0, -1].cpu().numpy(),
            "result": [] if len(result_temp) == 0 else result_temp[0][0],
            "reg_targets": data_agents["reg_targets"].cpu().numpy()[0],
            "anchors_map": data_agents["anchors"].cpu().numpy()[0],
            "gt_max_iou": data_agents["gt_max_iou"],
        }
        det_results_local[k], annotations_local[k] = cal_local_mAP(
            config, temp, det_results_local[k], annotations_local[k]
        )
    return det_results_local, annotations_local

def cw_l2_attack(model, inputs, labels, device, targeted=False, c=1e-4, kappa=0, max_iter=1000, learning_rate=0.01):
    # Define f-function
    def f(x):

        outputs = model(x)
        one_hot_labels = torch.eye(len(outputs[0]))[labels].to(device)

        i, _ = torch.max((1 - one_hot_labels) * outputs, dim=1)
        j = torch.masked_select(outputs, one_hot_labels.byte())

        # If targeted, optimize for making the other class most likely
        if targeted:
            return torch.clamp(i - j, min=-kappa)

        # If not targeted, optimize for making the other class most likely
        else:
            return torch.clamp(j - i, min=-kappa)

    w = torch.zeros_like(inputs, requires_grad=True).to(device)

    optimizer = optim.Adam([w], lr=learning_rate)

    prev = 1e10

    for step in range(max_iter):

        a = 1 / 2 * (nn.Tanh()(w) + 1)

        loss1 = nn.MSELoss(reduction='sum')(a, inputs)
        loss2 = torch.sum(c * f(a))

        cost = loss1 + loss2

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        # Early Stop when loss does not converge.
        if step % (max_iter // 10) == 0:
            if cost > prev:
                print('Attack Stopped due to CONVERGENCE....')
                return a
            prev = cost

        print('- Learning Progress : %2.2f %%        ' % ((step + 1) / max_iter * 100), end='\r')

    attack_inputs = 1 / 2 * (nn.Tanh()(w) + 1)

    return attack_inputs


def init(args):
    config = Config("test", binary=True, only_det=True)
    config_global = ConfigGlobal("train", binary=True, only_det=True)

    num_workers = args.nworker

    batch_size = args.batch

    # Specify gpu device
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    print(f"ego is using GPU 0")
    device_num = torch.cuda.device_count()
    print("device number", device_num)

    config.inference = args.inference
    # com, default: max
    if args.bound == "upperbound":
        flag = "upperbound"
    else:
        if args.com == "when2com":
            flag = "when2com"
            if args.inference == "argmax_test":
                flag = "who2com"
            if args.warp_flag:
                flag = flag + "_warp"
        elif args.com in {"v2v", "disco", "sum", "mean", "max", "cat", "agent"}:
            flag = args.com
        else:
            flag = "lowerbound"
            if args.box_com:
                flag += "_box_com"

    print("flag", flag)
    config.flag = flag

    num_agent = args.num_agent
    # `num_agent` exactly is the MAX num of agents
    agent_idx_range = range(1, num_agent) if args.no_cross_road else range(num_agent)
    validation_dataset = V2XSimDet(dataset_roots=[f"{args.data}/agent{i}" for i in agent_idx_range], config=config, config_global=config_global, split="val", val=True, bound=args.bound,
                                   kd_flag=args.kd_flag, no_cross_road=args.no_cross_road, )
    validation_data_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    if args.no_cross_road:
        num_agent -= 1

    start_epoch, fafmodule = init_model(config, args, num_agent, device)
    model_save_path = os.path.join(args.resume[: args.resume.rfind("/")])
    if args.inference == "argmax_test":
        model_save_path = model_save_path.replace("when2com", "who2com")
    os.makedirs(model_save_path, exist_ok=True)

    need_log = args.log
    log_file_path = os.path.join(args.logpath, "BTS_log")
    check_folder(log_file_path)
    log_file_name = os.path.join(log_file_path, f"epoch{start_epoch-1}_scene{args.scene_id}_ego{args.ego_agent}_{args.tolerant}_attackers_{args.pbft}_{time_str()}.log")
    root_logger = get_logger(log_file_name, need_log)
    root_logger.info(f"GPU number: {torch.cuda.device_count()}\n")
    # Logging the details for this experiment
    root_logger.info(f"command line: {' '.join(sys.argv[1:])}\n")
    root_logger.info(args.__repr__() + "\n\n")
    return config, num_agent, start_epoch, need_log, root_logger, fafmodule, model_save_path, agent_idx_range, validation_data_loader, device, batch_size


def get_logger(log_file_name, need_log):
    # Create a logger object
    logger = logging.getLogger("Root")
    # Set the logging level to INFO
    logger.setLevel(logging.INFO)
    # Generate log file name
    # Create a formatter to define the log output format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    # Create a stream handler to output logs to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    if need_log:
        # Create a file handler to write logs to a file
        file_handler = logging.FileHandler(log_file_name, mode="w")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def init_model(config, args, num_agent, device):
    # model: MaxFusion
    flag = config.flag
    compress_level = args.compress_level
    only_v2i = args.only_v2i
    if flag == "upperbound" or flag.startswith("lowerbound"):
        model = FaFNet(config, layer=args.layer, kd_flag=args.kd_flag, num_agent=num_agent)
    elif flag.startswith("when2com") or flag.startswith("who2com"):
        # model = PixelwiseWeightedFusionSoftmax(config, layer=args.layer)
        model = When2com(config, layer=args.layer, warp_flag=args.warp_flag, num_agent=num_agent, compress_level=compress_level, only_v2i=only_v2i, )
    elif args.com == "disco":
        model = DiscoNet(config, layer=args.layer, kd_flag=args.kd_flag, num_agent=num_agent, compress_level=compress_level, only_v2i=only_v2i, )
    elif args.com == "sum":
        model = SumFusion(config, layer=args.layer, kd_flag=args.kd_flag, num_agent=num_agent, compress_level=compress_level, only_v2i=only_v2i, )
    elif args.com == "mean":
        model = MeanFusion(config, layer=args.layer, kd_flag=args.kd_flag, num_agent=num_agent, compress_level=compress_level, only_v2i=only_v2i, )
    elif args.com == "max":
        model = MaxFusion(config, layer=args.layer, kd_flag=args.kd_flag, num_agent=num_agent, compress_level=compress_level, only_v2i=only_v2i, )
    elif args.com == "cat":
        model = CatFusion(config, layer=args.layer, kd_flag=args.kd_flag, num_agent=num_agent, compress_level=compress_level, only_v2i=only_v2i, )
    elif args.com == "agent":
        model = AgentWiseWeightedFusion(config, layer=args.layer, kd_flag=args.kd_flag, num_agent=num_agent, compress_level=compress_level, only_v2i=only_v2i, )
    else:
        model = V2VNet(config, gnn_iter_times=args.gnn_iter_times, layer=args.layer, layer_channel=256, num_agent=num_agent, compress_level=compress_level, only_v2i=only_v2i, )
    # model = nn.DataParallel(model)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = {"cls": SoftmaxFocalClassificationLoss(), "loc": WeightedSmoothL1LocalizationLoss(), }
    fafmodule = FaFModule(model, model, config, optimizer, criterion, args.kd_flag)
    checkpoint = torch.load(args.resume, map_location="cpu")  # We have low GPU utilization for testing
    start_epoch = checkpoint["epoch"] + 1
    # consume_prefix_in_state_dict_if_present()
    fafmodule.model.load_state_dict(checkpoint["model_state_dict"])
    fafmodule.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    fafmodule.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    print(f"Load model from {args.resume}, at epoch {start_epoch - 1}")
    return start_epoch, fafmodule


def mean_ap_evaluation(num_agent, args, det_results_local, annotations_local, root_logger, start_epoch):
    # local mAP evaluation
    mean_ap_local = []
    det_results_all_local = []
    annotations_all_local = []
    for k in range(num_agent):
        agent_k = k+1 if args.no_cross_road else k
        root_logger.info(f"Local mAP@0.5 from Agent {agent_k}(idx: {k})")
        mean_ap, _ = eval_map(det_results_local[k], annotations_local[k], scale_ranges=None, iou_thr=0.5, dataset=None, logger=root_logger, )
        mean_ap_local.append(mean_ap)
        root_logger.info(f"Local mAP@0.7 from Agent {agent_k}(idx: {k})")
        mean_ap, _ = eval_map(det_results_local[k], annotations_local[k], scale_ranges=None, iou_thr=0.7, dataset=None, logger=root_logger, )
        mean_ap_local.append(mean_ap)

        det_results_all_local += det_results_local[k]
        annotations_all_local += annotations_local[k]

    # average local mAP evaluation
    root_logger.info("Average Local mAP@0.5")
    mean_ap_local_average, _ = eval_map(det_results_all_local, annotations_all_local, scale_ranges=None, iou_thr=0.5, dataset=None, logger=root_logger, )
    mean_ap_local.append(mean_ap_local_average)

    root_logger.info("Average Local mAP@0.7")
    mean_ap_local_average, _ = eval_map(det_results_all_local, annotations_all_local, scale_ranges=None, iou_thr=0.7, dataset=None, logger=root_logger, )
    mean_ap_local.append(mean_ap_local_average)

    root_logger.info(f"Quantitative evaluation results of model from {args.resume}, at epoch {start_epoch - 1}")

    for k in range(num_agent):
        agent_k = k+1 if args.no_cross_road else k
        root_logger.info(f"Agent{agent_k}(idx: {k}) mAP@0.5 is {mean_ap_local[k * 2]} and mAP@0.7 is {mean_ap_local[(k * 2) + 1]}")

    root_logger.info(f"average local mAP@0.5 is {mean_ap_local[-2]} and average local mAP@0.7 is {mean_ap_local[-1]}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", default="/{your location}/dataset/CoSwarm-det/test", type=str, help="The path to the preprocessed sparse BEV training data", )
    parser.add_argument("--batch", default=1, type=int, help="The number of scene")
    parser.add_argument("--nepoch", default=50, type=int, help="Number of epochs")
    parser.add_argument("--nworker", default=10, type=int, help="Number of workers")
    parser.add_argument("--lr", default=0.001, type=float, help="Initial learning rate")
    parser.add_argument("--log", action="store_true", default=False, help="Whether to log")
    parser.add_argument("--logpath", default="/{your location}/BTS/coperception/logs", help="The path to the output log file")
    parser.add_argument("--resume", default="/{your location}/BTS/coperception/tools/det/runs/resume/max/with_cross/epoch_50.pth", type=str, help="The path to the saved model that is loaded to resume training", )
    parser.add_argument("--resume_teacher", default="", type=str, help="The path to the saved teacher model that is loaded to resume training", )
    parser.add_argument("--layer", default=3, type=int, help="Communicate which layer in the single layer com mode", )
    parser.add_argument("--warp_flag", action="store_true", help="Whether to use pose info for When2com")
    parser.add_argument("--kd_flag", default=0, type=int, help="Whether to enable distillation (only DiscNet is 1 )", )
    parser.add_argument("--kd_weight", default=100000, type=int, help="KD loss weight")
    parser.add_argument("--gnn_iter_times", default=3, type=int, help="Number of message passing for V2VNet", )
    parser.add_argument("--visualization", action="store_true", help="Visualize validation result")
    parser.add_argument("--com", default="max", type=str, help="disco/when2com/v2v/sum/mean/max/cat/agent")
    parser.add_argument("--bound", type=str, default="both", help="The input setting: lowerbound -> single-view or upperbound -> multi-view", )
    parser.add_argument("--inference", type=str)
    parser.add_argument("--box_com", action="store_true")
    parser.add_argument("--no_cross_road", action="store_true", help="Do not load data of cross roads")
    # scene_batch => batch size in each scene
    parser.add_argument("--num_agent", default=6, type=int, help="The total number of agents")
    parser.add_argument("--compress_level", default=0, type=int, help="Compress the communication layer channels by 2**x times in encoder", )
    parser.add_argument("--only_v2i", default=0, type=int, help="1: only v2i, 0: v2v and v2i", )
    # Adversarial perturbation
    parser.add_argument('--pert_alpha', type=float, default=0.1, help='scale of the perturbation')
    parser.add_argument('--adv_method', type=str, default='pgd', help='pgd/bim/cw-l2/noise_F/noise_P/noise_T')
    parser.add_argument('--eps', type=float, default=1.0, help='epsilon of adv attack.')
    parser.add_argument('--adv_iter', type=int, default=10
                        , help='adv iterations of computing perturbation')
    parser.add_argument("--noise_level",default=1.5, type=float, help="draw noise from normal distribution with given mean (in meters), apply to transformation matrix.")
    # Scene and frame settings
    parser.add_argument('--scene_id', type=int, nargs="+", default=[81], help='target evaluation scene, eg: --scene_id 8 96 97')  # Scene 8, 96, 97 has 6 agents.
    # [8,17,23,54,64] agent 4+1 [28,33,49,76,81] agent 7+1
    parser.add_argument('--sample_id', type=int, default=None, help='target evaluation sample')

    # pbft modes and parameters
    parser.add_argument('--pbft', type=str, default='pbft_mAP', help='upperbound/lowerbound/no_defense/pbft_mAP')
    parser.add_argument('--ego_agent', type=int, default=3, help='id of ego agent')
    parser.add_argument('--ego_loss_only', action="store_true", help='only use ego loss to compute adv perturbation')
    parser.add_argument('--box_matching_thresh', type=float, default=0.7, help='IoU threshold for validating two detection results')
    parser.add_argument('--tolerant', type=int, default=1, help='number of byzantine nodes are tolerant')
    parser.add_argument('--partial_upperbound', type=int, default=None, help='to perform clean collaboration with a subset of teammates')

    torch.multiprocessing.set_sharing_strategy("file_system")
    parsed_args = parser.parse_args()
    print(f"args: \n {parsed_args}")
    return parsed_args


def sus_check_jaccard(config, fafmodule, check_data, check_debug_times):
    data, num_agent, f, root_logger_dilled, jac_data, box_matching_thresh = check_data
    num_agent_list, gt_max_iou = jac_data.values()
    root_logger = dill.loads(root_logger_dilled)
    # no_fuse == None: return ego fuse with j(j==ego: ego-only)
    data["no_fuse"] = None

    st = time.time()
    with torch.no_grad():
        result = fafmodule.predict_all_in_ego(data, 1, num_agent=num_agent)
    check_debug_times.append(time.time() - st)
    print(f"check predict in {check_debug_times[-1]:.4f} seconds")
    # We use jaccard index to define the difference between two bbox sets
    reg_target = data["reg_targets"]
    padded_voxel_points = data["bev_seq"]
    anchors_map = data["anchors"]
    ego_idx = data["ego_agent"]
    jaccard = get_jaccard_only(config, ego_idx, num_agent_list, padded_voxel_points, reg_target, anchors_map, gt_max_iou, result)
    root_logger.info(f"jaccard values: {jaccard}")
    print(f"jaccard values: {jaccard}")
    jac_indices_le_thresh = [(i, jac) for i, jac in enumerate(jaccard) if jac <= box_matching_thresh]
    jac_indices_le_thresh.sort(key=lambda x: x[1])
    if len(jac_indices_le_thresh) == 0:
        root_logger.info(
            f"Ego {ego_idx}: No Agents have Jaccard coefficients less than the threshold {box_matching_thresh}.")
        return []
    min_f_jac = jac_indices_le_thresh[:f] if len(jac_indices_le_thresh) >= f else jac_indices_le_thresh
    sus_f_idx = [i[0] for i in min_f_jac]
    sus_f_jac = [i[1] for i in min_f_jac]

    root_logger.info(f"Ego {ego_idx} fused with suspicious Agent(s) {sus_f_idx} 's Jaccard Coefficient: {sus_f_jac}")
    return sus_f_idx
