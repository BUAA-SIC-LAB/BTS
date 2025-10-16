from tqdm import tqdm
from BTS_util import *
from coperception.utils.data_util import add_binary_noise
from tools.det.BTS.pbft_multiprocess_vote import PBFTProtocol

# debug
os.environ["RDMAV_FORK_SAFE"] = "1"

PBFTMODE = ["pbft_mAP", "lowerbound", "upperbound", "no_defense"]


# @torch.no_grad()
# We cannot use torch.no_grad() since we need to calculate the gradient for perturbation1
def main(args):
    random.seed(0)
    st = time.time()
    assert args.pbft in PBFTMODE
    config, num_agent, start_epoch, need_log, root_logger, fafmodule, model_save_path, agent_idx_range, validation_data_loader, device, batch_size = init(args)
    ego_idx = args.ego_agent - 1 if args.no_cross_road else args.ego_agent
    print(f"ego_agent: {args.ego_agent}")
    # `all_agent_list` is the data idx of agents, which not means the origin Serial Number
    all_agent_list = [i-1 if args.no_cross_road else i for i in agent_idx_range]
    # We always trust ego
    all_agent_list.remove(ego_idx)
    print(f"all_agent_list: {all_agent_list}") # [0, 2, 3, 4]

    # init pbft server
    f = args.tolerant
    # primary Node (Agent 0)
    primary = 0
    print(f"primary: {primary}")
    if args.pbft == "pbft_mAP":
        pbft = PBFTProtocol(config, args, num_agent, f, all_agent_list, byzantine_ids=[], primary=primary)
        pbft.init()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    #  ===== eval =====
    fafmodule.model.eval()
    save_fig_path = [check_folder(os.path.join(args.logpath, "vis_pbft/", f"vis{i}")) for i in agent_idx_range]

    # for local and global mAP evaluation
    det_results_local = [[] for _ in agent_idx_range]
    annotations_local = [[] for _ in agent_idx_range]
    # fix parameters
    for k, v in fafmodule.model.named_parameters():
        v.requires_grad = False

    # ego INIT
    # counters for relative frame in a single scene
    frame_seq = 0
    iter_times = []
    num_success = 0
    for cnt, sample in enumerate(tqdm(validation_data_loader)):
        iter_start = time.time()
        (padded_voxel_point_list, padded_voxel_points_teacher_list, label_one_hot_list, reg_target_list, reg_loss_mask_list, anchors_map_list, vis_maps_list, gt_max_iou, filenames,
         target_agent_id_list, num_agent_list, trans_matrices_list) = zip(*sample)

        filename0 = filenames[0]
        filename = str(filename0[0][0])
        cut = filename[filename.rfind('agent') + 7:]
        seq_name = cut[:cut.rfind('_')]
        idx = cut[cut.rfind('_') + 1:cut.rfind('/')]

        # ignore unspecified scene_id
        if int(seq_name) not in args.scene_id:
            continue
        # ignore unspecified sample_id
        if args.sample_id is not None:
            if int(idx) not in args.sample_id:
                continue
        frame_seq += 1
        root_logger.info("Scene {}, Frame {}:".format(seq_name, idx))
        trans_matrices = torch.stack(tuple(trans_matrices_list), 1)

        target_agent_ids = torch.stack(tuple(target_agent_id_list), 1)
        num_all_agents = torch.stack(tuple(num_agent_list), 1)
        if args.no_cross_road:
            num_all_agents -= 1

        padded_voxel_points = torch.cat(tuple(padded_voxel_point_list), 0)
        padded_voxel_points_teacher = torch.cat(tuple(padded_voxel_points_teacher_list), 0)
        label_one_hot = torch.cat(tuple(label_one_hot_list), 0)
        reg_target = torch.cat(tuple(reg_target_list), 0)
        reg_loss_mask = torch.cat(tuple(reg_loss_mask_list), 0)
        anchors_map = torch.cat(tuple(anchors_map_list), 0)
        vis_maps = torch.cat(tuple(vis_maps_list), 0)

        unadv_pert = None

        data = {"bev_seq": padded_voxel_points.to(device), "bev_seq_teacher": padded_voxel_points_teacher.to(device), "labels": label_one_hot.to(device), "reg_targets": reg_target.to(device),
                "anchors": anchors_map.to(device), "vis_maps": vis_maps.to(device), "reg_loss_mask": reg_loss_mask.to(device).type(dtype=torch.bool), "target_agent_ids": target_agent_ids.to(device),
                "num_agent": num_all_agents.to(device), "ego_agent": ego_idx, "pert": None, "no_fuse": False, "collab_agent_list": None, 'trial_agent_id': None, 'confidence': None,
                'unadv_pert': unadv_pert, "attacker_list": None, "eps": None, "trans_matrices": trans_matrices.to(device), }

        # cal_forward_time
        if args.pbft == "performance_eval":
            fafmodule.cal_forward_time(data, 1)
            continue
        # 1. ego REQUEST
        # 1.1 get original ego agent class prediction of all anchors, without adv pert and fuse, return cls pred of all agents
        if args.pbft != "lowerbound" or args.pbft != "upperbound":
            cls_result = fafmodule.cls_predict(data, batch_size, no_fuse=True)
            # change logits to one-hot.
            mean = torch.mean(cls_result, dim=2)
            cls_result[:, :, 0] = cls_result[:, :, 0] > mean
            cls_result[:, :, 1] = cls_result[:, :, 1] > mean
            pseudo_gt = cls_result.clone().detach()
        # 1.2 BTS mode: upperbound/lowerbound/pbft_mAP/no_defence
        # all agents collaborate
        if args.pbft == 'upperbound':
            data['pert'] = None
            if args.partial_upperbound is not None:
                # Sometimes we need to eval partially colloborative agents
                collab_agent_list = random.sample([i for i in all_agent_list if i != 0], k=args.partial_upperbound)
                collab_agent_list.append(0)
                # collab_agent_list = [all_agent_list[args.partial_upperbound]]
                data['collab_agent_list'] = collab_agent_list
                root_logger.info(f"Partial upperbound, collab agent list: {collab_agent_list}")
            else:
                data['collab_agent_list'] = None
            data['trial_agent_id'] = None
            data['no_fuse'] = False
            loss, cls_loss, loc_loss, result = fafmodule.predict_all(data, 1, num_agent=num_agent)
            if args.visualization:
                visualize(config, args.visualization, filename0, save_fig_path, fafmodule, data, num_agent_list, gt_max_iou, vis_tag='upperbound')
            det_results_local, annotations_local = local_eval(num_agent, padded_voxel_points, reg_target, anchors_map, gt_max_iou, result, config, det_results_local, annotations_local)
            continue
        # Suppose all neighboring agents are malicious, and only the ego agent is trusted
        # Each agent only use its own features to perform object detection
        elif args.pbft == 'lowerbound':
            data['pert'] = None
            data['collab_agent_list'] = None
            data['trial_agent_id'] = None
            data['no_fuse'] = True
            loss, cls_loss, loc_loss, result = fafmodule.predict_all(data, 1, num_agent=num_agent)
            if args.visualization:
                # visualize attacked result
                visualize(config, args.visualization, filename0, save_fig_path, fafmodule, data, num_agent_list, gt_max_iou, vis_tag='lowerbound')
            det_results_local, annotations_local = local_eval(num_agent, padded_voxel_points, reg_target, anchors_map, gt_max_iou, result, config, det_results_local, annotations_local)
            continue
        # There are attackers, ego needs to find them:
        else:  # attack happened
            # 1.3 generate adv perturb
            if args.adv_method == 'pgd':
                # PGD random init
                pert = torch.randn(num_agent, 256, 32, 32) * args.eps
            elif args.adv_method == 'bim' or args.adv_method == 'cw-l2':
                # BIM/CW-L2 zero init
                pert = torch.zeros(num_agent, 256, 32, 32)
            elif args.adv_method == 'noise_F':
                # noise on Feature init
                pert = torch.randn(num_agent, 256, 32, 32) * args.noise_level
                unadv_pert = True
            elif args.adv_method == 'noise_P':
                # noise on Point init
                unadv_pert = False
                pert = None
            elif args.adv_method == 'noise_T':
                pert = None
                # add pose noise
                unadv_pert = args.noise_level
            else:
                raise NotImplementedError

            # Randomly samples neighboring agents as attackers
            # temporarily remove primary Node
            if args.pbft == 'pbft_mAP':
                all_agent_list_candidate = [i for i in all_agent_list if i != pbft.primary]
            else:
                all_agent_list_candidate = all_agent_list
            attacker_list = random.sample(all_agent_list_candidate, k=args.tolerant)
            data['attacker_list'] = attacker_list
            data['eps'] = args.eps
            data['no_fuse'] = False
            # visualize LB det and IUB result
            if args.visualization:
                # visualize UB det result, without fusion
                data['no_fuse'] = True
                visualize(config, args.visualization, filename0, save_fig_path, fafmodule, data, num_agent_list,
                          gt_max_iou, vis_tag='LB')
                # visualize IUB result, fusion with no attack
                data['no_fuse'] = False
                visualize(config, args.visualization, filename0, save_fig_path, fafmodule, data, num_agent_list,
                          gt_max_iou, vis_tag='IUB')
            if args.adv_method != 'noise_F' and args.adv_method != 'noise_P' and args.adv_method != 'noise_T':
                # Introduce adv perturbation
                for i in range(args.adv_iter):
                    pert.requires_grad = True
                    data['pert'] = pert.to(device)
                    # STEP 3: Use inverted classification ground truth, minimize loss wrt inverted gt, to generate adv attacks based on cls(only)
                    # NOTE: Actual ground truth is not always available especially in real-world attacks
                    # We define the adversarial loss of the perturbed output with respect to an unperturbed output pseudo_gt instead of the ground truth
                    _ = fafmodule.cls_step(data, batch_size, ego_loss_only=args.ego_loss_only, ego_agent=ego_idx, invert_gt=True, self_result=pseudo_gt, adv_method=args.adv_method)
                    pert = pert + args.pert_alpha * pert.grad.sign() * -1
                    pert.detach_()
                # Detach and clone perturbations from Pytorch computation graph, in case of gradient misuse.
                pert = pert.detach().clone()

            # Apply the final perturbation to attackers' feature maps.
            data['pert'] = pert.to(device) if pert is not None else None
            root_logger.info(f"Perturbation is applied on agent {attacker_list}")
            # visualize ND and UB
            if args.visualization:
                data["no_fuse"] = False
                # ND, with attack and fusion
                visualize(config, args.visualization, filename0, save_fig_path, fafmodule, data, num_agent_list, gt_max_iou,
                          vis_tag='ND')
                data['pert'] = None
                data['collab_agent_list'] = list(set(all_agent_list) - set(attacker_list))
                # UB, with oracle to exclude the attack
                visualize(config, args.visualization, filename0, save_fig_path, fafmodule, data, num_agent_list, gt_max_iou,
                          vis_tag='UB')

            if args.pbft == 'no_defense':
                data['no_fuse'] = False
                loss, cls_loss, loc_loss, result = fafmodule.predict_all(data, 1, num_agent=num_agent)
                det_results_local, annotations_local = local_eval(num_agent, padded_voxel_points, reg_target, anchors_map, gt_max_iou, result, config, det_results_local, annotations_local)
                continue

            if args.pbft == "pbft_mAP":
                found = False
                request_msg = f"Scene{seq_name} Frame{cnt} request"
                data_no_device = {"bev_seq": add_binary_noise(padded_voxel_points, attacker_list, args.noise_level) if not unadv_pert else padded_voxel_points,
                                  "bev_seq_teacher": padded_voxel_points_teacher,
                                  "labels": label_one_hot, "reg_targets": reg_target, "anchors": anchors_map, "vis_maps": vis_maps,
                                  "reg_loss_mask": reg_loss_mask.type(dtype=torch.bool), "target_agent_ids": target_agent_ids,
                                  "num_agent": num_all_agents, "trans_matrices": trans_matrices, "pert": pert, # need to(device)
                                  "ego_agent": ego_idx, "no_fuse": None, "collab_agent_list": None,
                                  'trial_agent_id': None, 'confidence': None, 'unadv_pert': unadv_pert, "attacker_list": attacker_list, "eps": args.eps}
                jac_data = {"num_agent_list": num_agent_list, "gt_max_iou": gt_max_iou,}
                root_logger_dilled = dill.dumps(root_logger)
                other_data = {"num_agent": num_agent, "root_logger_dilled":root_logger_dilled}

                pbft.update_pbft(attacker_list, data_no_device, other_data, jac_data)
                # run the PBFT protocol
                reply, sus_agent_list = pbft.run(request_msg)
                if sus_agent_list == set(attacker_list):
                    num_success += 1
                    found = True
                reply_agent = [i + 1 for i in reply] if args.no_cross_road else reply
                if found:
                    root_logger.info(f"ego Agent {args.ego_agent}(idx: {ego_idx}) make consensus with Agents {reply_agent}(idx:{reply})!")
                    # Then ego make fusion after eliminating attackers
                    collab_agent_list = list(set(all_agent_list) - set(sus_agent_list))
                    # data['pert'] = None # If found correctly, even if there is the `pert`, it won't take effect
                    data['collab_agent_list'] = collab_agent_list
                    data['no_fuse'] = False
                    root_logger.info(f"Ego think Agent(s) {sus_agent_list} was attacked, and will fuse with Agent idx {collab_agent_list}")
                else:
                    # predict ego only
                    data['pert'] = None
                    data['collab_agent_list'] = None
                    data['no_fuse'] = True
                    root_logger.info(f"PBFT got no consensus!, ego will predict only")
                fuse_times = time.time()
                loss, cls_loss, loc_loss, result = fafmodule.predict_all(data, 1, num_agent=num_agent)
                root_logger.info(f"fuse time: {time.time() - fuse_times}")
                det_results_local, annotations_local = local_eval(num_agent, padded_voxel_points, reg_target, anchors_map, gt_max_iou, result, config, det_results_local, annotations_local)
                if args.visualization:
                    visualize(config, args.visualization, filename0, save_fig_path, fafmodule, data, num_agent_list,
                          gt_max_iou, vis_tag='BTS')
        iter_end = time.time()
        iter_times.append(iter_end - iter_start)
    if args.pbft == "pbft_mAP":
        pbft.clean()
        root_logger.info(f"PBFT nodes clean up!")
    root_logger.info(f"Ego Agent:{args.ego_agent}(idx: {ego_idx})")

    if args.pbft != 'lowerbound' or args.pbft != 'upperbound':
        root_logger.info(f"pbft VALIDATION: Evaluated on {frame_seq} frames")
        root_logger.info(f"Total Neighbor Agents:{num_agent - 1}")
        root_logger.info(f"Box set matching threshold: {args.box_matching_thresh}")
    # mAP evaluation (local and global)
    mean_ap_evaluation(num_agent, args, det_results_local, annotations_local, root_logger, start_epoch)
    root_logger.info(f"success times: {num_success}")
    # predict agents-only avg time
    root_logger.info(f"frame iter avg time: {np.mean(iter_times):.4f} s")
    # pbft protocol avg time
    if args.pbft == "pbft_mAP":
        root_logger.info(f"pbft protocol avg time: {np.mean(pbft.avg_times):.4f} s")
    root_logger.info(f"total time: {time.time()-st:.4f} s")


if __name__ == "__main__":
    # mp.set_start_method('spawn', force=True)
    main_args = get_args()
    main(main_args)
