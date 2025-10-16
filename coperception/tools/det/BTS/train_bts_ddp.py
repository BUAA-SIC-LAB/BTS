"""
modified by Zhenghao Liu
training by DDP instead of DP
"""
import argparse
import logging
import random

import torch.optim as optim
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from coperception.datasets import V2XSimDet as CoSwarmDet
from coperception.configs import Config, ConfigGlobal
from coperception.utils.CoDetModule import *
from coperception.utils.data_util import apply_pose_noise
from coperception.utils.loss import *
from coperception.models.det import *
from coperception.utils import AverageMeter

import glob
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
coperception_dir = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
sys.path.append(coperception_dir)
from tools.det.BTS.BTS_util import check_folder

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    init_seeds(1+rank,False)
    torch.cuda.set_device(rank)
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def get_logger(log_file_name, rank, need_log, mode="w"):
    # Create a logger object
    logger = logging.getLogger("trainer")
    # Set the logging level to INFO
    level = logging.INFO if rank in [-1, 0] else logging.WARN
    logger.setLevel(level)
    # Generate log file name
    # Create a formatter to define the log output format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    # Create a stream handler to output logs to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    if need_log:
        # Create a file handler to write logs to a file
        file_handler = logging.FileHandler(log_file_name, mode=mode)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger

def init_seeds(seed=0, cuda_deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True

def cleanup():
    dist.destroy_process_group()

def validate(faf, dataloader, rank, epoch, batch_size=1):
    faf.model.eval()
    val_loss_disp = AverageMeter("Total loss", ":.6f")
    val_loss_class = AverageMeter("classification Loss", ":.6f")  # for cell classification error
    val_loss_loc = AverageMeter("Localization Loss", ":.6f")  # for state estimation error

    with torch.no_grad():
        t = tqdm(dataloader, disable=rank != 0)
        for sample in t:
            (padded_voxel_point_list, padded_voxel_points_teacher_list, label_one_hot_list, reg_target_list,
             reg_loss_mask_list, anchors_map_list, vis_maps_list, _, _,
             target_agent_id_list, num_agent_list, trans_matrices_list) = zip(*sample)
            trans_matrices = torch.stack(tuple(trans_matrices_list), 1)
            target_agent_id = torch.stack(tuple(target_agent_id_list), 1)
            num_all_agents = torch.stack(tuple(num_agent_list), 1)
            padded_voxel_point = torch.cat(tuple(padded_voxel_point_list), 0)
            # padded_voxel_points_teacher = torch.cat(tuple(padded_voxel_points_teacher_list), 0)
            label_one_hot = torch.cat(tuple(label_one_hot_list), 0)
            reg_target = torch.cat(tuple(reg_target_list), 0)
            reg_loss_mask = torch.cat(tuple(reg_loss_mask_list), 0)
            anchors_map = torch.cat(tuple(anchors_map_list), 0)
            vis_maps = torch.cat(tuple(vis_maps_list), 0)
            data = {"bev_seq": padded_voxel_point.to(rank), "labels": label_one_hot.to(rank),
                    "reg_targets": reg_target.to(rank), "anchors": anchors_map.to(rank),
                    "reg_loss_mask": reg_loss_mask.to(rank).type(dtype=torch.bool), "vis_maps": vis_maps.to(rank),
                    "target_agent_ids": target_agent_id.to(rank), "num_agent": num_all_agents.to(rank),
                    "trans_matrices": trans_matrices, }

            loss, cls_loss, loc_loss = faf.step_val(data, batch_size, num_agent=num_all_agents[0][0])
            val_loss_disp.update(loss)
            val_loss_class.update(cls_loss)
            val_loss_loc.update(loc_loss)
            t.set_description(f"Val in Epoch: {epoch} ")
            t.set_postfix(cls_loss=val_loss_class.avg, loc_loss=val_loss_loc.avg)
            torch.cuda.empty_cache()
    return val_loss_disp.avg, val_loss_class.avg, val_loss_loc.avg

def main(rank, world_size, args):
    setup(rank, world_size)

    config = Config("train", binary=True, only_det=True)
    config_global = ConfigGlobal("train", binary=True, only_det=True)

    num_epochs = args.nepoch
    need_log = args.log
    num_workers = args.nworker
    start_epoch = 1
    batch_size = args.batch
    compress_level = args.compress_level
    auto_resume_path = args.auto_resume_path
    pose_noise = args.pose_noise
    only_v2i = args.only_v2i

    # Specify gpu device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_num = torch.cuda.device_count()
    print(f"rank {rank}: device number", device_num)

    if args.bound == "upperbound":
        flag = "upperbound"
    elif args.bound == "lowerbound":
        if args.com == "when2com" and args.warp_flag:
            flag = "when2com_warp"
        elif args.com in {"v2v", "disco", "sum", "mean", "max", "cat", "agent", "when2com", }:
            flag = args.com
        else:
            flag = "lowerbound"
    else:
        raise ValueError("not implement")

    config.flag = flag

    num_agent = args.num_agent
    # agent0 is the cross road
    agent_idx_range = range(1, num_agent) if args.no_cross_road else range(num_agent)
    training_dataset = CoSwarmDet(dataset_roots=[f"{args.data}/train/agent{i}" for i in agent_idx_range], config=config,
                                 config_global=config_global, split="train", bound=args.bound, kd_flag=args.kd_flag,
                                 no_cross_road=args.no_cross_road, )
    training_sampler = DistributedSampler(training_dataset, num_replicas=world_size, rank=rank,shuffle=True)
    training_data_loader = DataLoader(training_dataset, batch_size=batch_size, num_workers=num_workers, sampler=training_sampler)
    val_dataset = CoSwarmDet(dataset_roots=[f"{args.data}/val/agent{i}" for i in agent_idx_range], config=config,
                                 config_global=config_global, split="val", val=True, bound=args.bound, kd_flag=args.kd_flag,
                                 no_cross_road=args.no_cross_road, )
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank,shuffle=False)
    val_data_loader = DataLoader(val_dataset, batch_size=1, num_workers=num_workers,sampler=val_sampler)

    print("Training dataset size:", len(training_dataset))
    print("val dataset size:", len(val_dataset))

    logger_root = args.logpath if args.logpath != "" else "logs"


    if args.no_cross_road:
        num_agent -= 1

    if args.com == "":
        model = FaFNet(config, layer=args.layer, kd_flag=args.kd_flag, num_agent=num_agent,
                       compress_level=compress_level, )
    elif args.com == "when2com":
        model = When2com(config, layer=args.layer, warp_flag=args.warp_flag, num_agent=num_agent,
                         compress_level=compress_level, only_v2i=only_v2i, )
    elif args.com == "v2v":
        model = V2VNet(config, gnn_iter_times=args.gnn_iter_times, layer=args.layer, layer_channel=256,
                       num_agent=num_agent, compress_level=compress_level, only_v2i=only_v2i, )
    elif args.com == "disco":
        model = DiscoNet(config, layer=args.layer, kd_flag=args.kd_flag, num_agent=num_agent,
                         compress_level=compress_level, only_v2i=only_v2i, )
    elif args.com == "sum":
        model = SumFusion(config, layer=args.layer, kd_flag=args.kd_flag, num_agent=num_agent,
                          compress_level=compress_level, only_v2i=only_v2i, )
    elif args.com == "mean":
        model = MeanFusion(config, layer=args.layer, kd_flag=args.kd_flag, num_agent=num_agent,
                           compress_level=compress_level, only_v2i=only_v2i, )
    elif args.com == "max":
        model = MaxFusion(config, layer=args.layer, kd_flag=args.kd_flag, num_agent=num_agent,
                          compress_level=compress_level, only_v2i=only_v2i, )
    elif args.com == "cat":
        model = CatFusion(config, layer=args.layer, kd_flag=args.kd_flag, num_agent=num_agent,
                          compress_level=compress_level, only_v2i=only_v2i, )
    elif args.com == "agent":
        model = AgentWiseWeightedFusion(config, layer=args.layer, kd_flag=args.kd_flag, num_agent=num_agent,
                                        compress_level=compress_level, only_v2i=only_v2i, )
    else:
        raise NotImplementedError("Invalid argument com:" + args.com)

    # model = model.to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
    # model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    model = DDP(model, device_ids=[rank])

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = {"cls": SoftmaxFocalClassificationLoss(), "loc": WeightedSmoothL1LocalizationLoss(), }

    if args.kd_flag == 1:
        teacher = TeacherNet(config)
        teacher = nn.DataParallel(teacher)
        teacher = teacher.to(device)
        faf_module = FaFModule(model, teacher, config, optimizer, criterion, args.kd_flag)
        checkpoint_teacher = torch.load(args.resume_teacher)
        start_epoch_teacher = checkpoint_teacher["epoch"]
        faf_module.teacher.load_state_dict(checkpoint_teacher["model_state_dict"])
        print("Load teacher model from {}, at epoch {}".format(args.resume_teacher, start_epoch_teacher))
        faf_module.teacher.eval()
    else:
        faf_module = FaFModule(model, model, config, optimizer, criterion, args.kd_flag)

    cross_path = "no_cross" if args.no_cross_road else "with_cross"
    model_save_path = check_folder(logger_root)
    model_save_path = check_folder(os.path.join(model_save_path, flag))
    # 初始化SummaryWriter
    writer = SummaryWriter(log_dir=model_save_path)
    if args.no_cross_road:
        model_save_path = check_folder(os.path.join(model_save_path, "no_cross"))
    else:
        model_save_path = check_folder(os.path.join(model_save_path, "with_cross"))

    # check if there is valid check point file
    has_valid_pth = False
    for pth_file in os.listdir(check_folder(os.path.join(auto_resume_path, f"{flag}/{cross_path}"))):
        if pth_file.startswith("epoch_") and pth_file.endswith(".pth"):
            has_valid_pth = True
            break

    if not has_valid_pth:
        print(f"No valid check point file in {auto_resume_path} dir, weights not loaded.")
        auto_resume_path = ""

    if args.resume == "" and auto_resume_path == "":
        log_file_name = os.path.join(model_save_path, "log.txt")
        logger = get_logger(log_file_name,rank, need_log)
    else:
        if auto_resume_path != "":
            model_save_path = os.path.join(auto_resume_path, f"{flag}/{cross_path}")
        else:
            model_save_path = args.resume[: args.resume.rfind("/")]
        print(f"model save path: {model_save_path}")
        log_file_name = os.path.join(model_save_path, "log.txt")
        if os.path.exists(log_file_name):
            logger = get_logger(log_file_name, rank, need_log,mode="a")
        else:
            os.makedirs(model_save_path, exist_ok=True)
            logger = get_logger(log_file_name, rank, need_log)
        logger.info("GPU number: {}\n".format(torch.cuda.device_count()))
        # Logging the details for this experiment
        logger.info("command line: {}\n".format(" ".join(sys.argv[0:])))
        logger.info(args.__repr__() + "\n\n")

        if auto_resume_path != "":
            list_of_files = glob.glob(f"{model_save_path}/*.pth")
            latest_pth = max(list_of_files, key=os.path.getctime)
            checkpoint = torch.load(latest_pth)
        else:
            checkpoint = torch.load(args.resume)

        start_epoch = checkpoint["epoch"] + 1
        model_state_dict = {f"module.{key}": value for key, value in checkpoint["model_state_dict"].items()}
        faf_module.model.load_state_dict(model_state_dict)
        faf_module.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        faf_module.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        print("Load model from {}, at epoch {}".format(args.resume, start_epoch - 1))
    for epoch in range(start_epoch, num_epochs + 1):
        training_sampler.set_epoch(epoch)
        lr = faf_module.optimizer.param_groups[0]["lr"]
        # print("Epoch {}, learning rate {}".format(epoch, lr))
        logger.info("epoch: {}, lr: {}\t".format(epoch, lr))

        running_loss_disp = AverageMeter("Total loss", ":.6f")
        running_loss_class = AverageMeter("classification Loss", ":.6f")  # for cell classification error
        running_loss_loc = AverageMeter("Localization Loss", ":.6f")  # for state estimation error

        faf_module.model.train()

        t = tqdm(training_data_loader, dynamic_ncols=True, disable=rank != 0)
        for sample in t:
            (padded_voxel_point_list, padded_voxel_points_teacher_list, label_one_hot_list, reg_target_list,
             reg_loss_mask_list, anchors_map_list, vis_maps_list, target_agent_id_list, num_agent_list,
             trans_matrices_list,) = zip(*sample)

            trans_matrices = torch.stack(tuple(trans_matrices_list), 1)
            target_agent_id = torch.stack(tuple(target_agent_id_list), 1)
            num_all_agents = torch.stack(tuple(num_agent_list), 1)

            # add pose noise
            if pose_noise > 0:
                apply_pose_noise(pose_noise, trans_matrices)
            if args.no_cross_road:
                num_all_agents -= 1
            if flag == "upperbound":
                padded_voxel_point = torch.cat(tuple(padded_voxel_points_teacher_list), 0)
            else:
                padded_voxel_point = torch.cat(tuple(padded_voxel_point_list), 0)

            label_one_hot = torch.cat(tuple(label_one_hot_list), 0)
            reg_target = torch.cat(tuple(reg_target_list), 0)
            reg_loss_mask = torch.cat(tuple(reg_loss_mask_list), 0)
            anchors_map = torch.cat(tuple(anchors_map_list), 0)
            vis_maps = torch.cat(tuple(vis_maps_list), 0)

            data = {"bev_seq": padded_voxel_point.to(device), "labels": label_one_hot.to(device),
                    "reg_targets": reg_target.to(device), "anchors": anchors_map.to(device),
                    "reg_loss_mask": reg_loss_mask.to(device).type(dtype=torch.bool), "vis_maps": vis_maps.to(device),
                    "target_agent_ids": target_agent_id.to(device), "num_agent": num_all_agents.to(device),
                    "trans_matrices": trans_matrices, }

            if args.kd_flag == 1:
                padded_voxel_points_teacher = torch.cat(tuple(padded_voxel_points_teacher_list), 0)
                data["bev_seq_teacher"] = padded_voxel_points_teacher.to(device)
                data["kd_weight"] = args.kd_weight

            loss, cls_loss, loc_loss = faf_module.step(data, batch_size, num_agent=num_agent)
            running_loss_disp.update(loss)
            running_loss_class.update(cls_loss)
            running_loss_loc.update(loc_loss)

            if np.isnan(loss) or np.isnan(cls_loss) or np.isnan(loc_loss):
                print(f"Epoch {epoch}, loss is nan: {loss}, {cls_loss} {loc_loss}")
                sys.exit()

            t.set_description("Epoch {},     lr {}".format(epoch, lr))
            t.set_postfix(cls_loss=running_loss_class.avg, loc_loss=running_loss_loc.avg)
        # 记录每个epoch的训练损失
        writer.add_scalar('Train Total Loss', running_loss_disp.avg, epoch)
        writer.add_scalar('Train Classification Loss', running_loss_class.avg, epoch)
        writer.add_scalar('Train Localization Loss', running_loss_loc.avg, epoch)

        # Record the validation loss every 2 epochs
        if (epoch-1) % 2 == 0:
            loss, cls_loss, loc_loss = validate(faf_module,val_data_loader, rank, epoch,1)
            writer.add_scalar('Val Total Loss', loss, epoch)
            writer.add_scalar('Val Classification Loss', cls_loss, epoch)
            writer.add_scalar('Val Localization Loss', loc_loss, epoch)

        faf_module.scheduler.step()
        # save lr
        writer.add_scalar('Learning Rate', faf_module.scheduler.get_last_lr()[0], epoch)
        logger.info("{}\t{}\t{}\n".format(running_loss_disp, running_loss_class, running_loss_loc))
        # save model
        if rank == 0:
            if config.MGDA:
                save_dict = {"epoch": epoch, "encoder_state_dict": faf_module.encoder.state_dict(),
                             "optimizer_encoder_state_dict": faf_module.optimizer_encoder.state_dict(),
                             "scheduler_encoder_state_dict": faf_module.scheduler_encoder.state_dict(),
                             "head_state_dict": faf_module.head.state_dict(),
                             "optimizer_head_state_dict": faf_module.optimizer_head.state_dict(),
                             "scheduler_head_state_dict": faf_module.scheduler_head.state_dict(),
                             "loss": running_loss_disp.avg, }
            else:
                # changed model.module
                save_dict = {"epoch": epoch, "model_state_dict": faf_module.model.module.state_dict(),
                             "optimizer_state_dict": faf_module.optimizer.state_dict(),
                             "scheduler_state_dict": faf_module.scheduler.state_dict(), "loss": running_loss_disp.avg, }
            torch.save(save_dict, os.path.join(model_save_path, "epoch_" + str(epoch) + ".pth"))
        # dist.barrier()
    # Close SummaryWriter
    writer.close()
    # clear dist
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", default="/{your location}/dataset/CoSwarm-det",
        type=str, help="The path to the preprocessed sparse BEV training data", )
    parser.add_argument("--batch", default=4, type=int, help="Batch size")
    parser.add_argument("--nepoch", default=50, type=int, help="Number of epochs")
    parser.add_argument("--nworker", default=4, type=int, help="Number of workers")
    parser.add_argument("--lr", default=0.001, type=float, help="Initial learning rate")
    parser.add_argument("--log", action="store_true",default=True, help="Whether to log")
    parser.add_argument("--logpath", default="/{your location}/BTS/coperception/tools/det/runs/train/ddp", help="The path to the output log file")
    parser.add_argument("--resume", default="", type=str,
        help="The path to the saved model that is loaded to resume training", )
    parser.add_argument("--resume_teacher", default="", type=str,
        help="The path to the saved teacher model that is loaded to resume training", )
    parser.add_argument("--layer", default=3, type=int, help="Communicate which layer in the single layer com mode", )
    parser.add_argument("--warp_flag", action="store_true", help="Whether to use pose info for Ｗhen2com")
    parser.add_argument("--kd_flag", default=0, type=int, help="Whether to enable distillation (only DiscNet is 1 )", )
    parser.add_argument("--kd_weight", default=100000, type=int, help="KD loss weight")
    parser.add_argument("--gnn_iter_times", default=3, type=int, help="Number of message passing for V2VNet", )
    parser.add_argument("--visualization", default=False, help="Visualize validation result")
    # parser.add_argument("--com", default="mean", type=str, help="disco/when2com/v2v/sum/mean/max/cat/agent")
    parser.add_argument("--com", default="max", type=str, help="disco/when2com/v2v/sum/mean/max/cat/agent")
    parser.add_argument("--bound", type=str, default="lowerbound",
        help="The input setting: lowerbound -> single-view or upperbound -> multi-view", )
    parser.add_argument("--no_cross_road", action="store_true", help="Do not load data of cross roads")
    parser.add_argument("--num_agent", default=8, type=int, help="The total number of agents")
    parser.add_argument("--auto_resume_path", default="/{your location}/BTS/coperception/tools/det/runs/train/ddp", type=str,
        help="The path to automatically reload the latest pth", )
    parser.add_argument("--compress_level", default=0, type=int,
        help="Compress the communication layer channels by 2**x times in encoder", )
    parser.add_argument("--pose_noise", default=0, type=float,
        help="draw noise from normal distribution with given mean (in meters), apply to transformation matrix.", )
    parser.add_argument("--only_v2i", default=0, type=int, help="1: only v2i, 0: v2v and v2i", )

    torch.multiprocessing.set_sharing_strategy("file_system")
    args = parser.parse_args()
    print(args)

    world_size = torch.cuda.device_count()
    # world_size = 2
    mp.spawn(main, args=(world_size, args), nprocs=world_size, join=True)
