import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval, R1_mAP
from torch.cuda import amp
import torch.distributed as dist
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import logging

def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS
    logging.getLogger().setLevel(logging.INFO)
    logger = logging.getLogger("TOPReID.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                              find_unused_parameters=True)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    if cfg.DATASETS.NAMES == "MSVR310":
        evaluator = R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    else:
        evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    # train
    best_index = {'mAP': 0, "Rank-1": 0, 'Rank-5': 0, 'Rank-10': 0}
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)
        model.train()
        for n_iter, (img, vid, target_cam, target_view, _) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = {'RGB': img['RGB'].to(device),
                   'NI': img['NI'].to(device),
                   'TI': img['TI'].to(device)}
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            with amp.autocast(enabled=True):
                output = model(img, label=target, cam_label=target_cam, view_label=target_view)
                loss = 0
                if cfg.MODEL.RE:
                    index = len(output) - 1
                    for i in range(0, index, 2):
                        loss_tmp = loss_fn(score=output[i], feat=output[i + 1], target=target, target_cam=target_cam)
                        loss = loss + loss_tmp
                    loss = loss + output[-1]
                else:
                    index = len(output)
                    for i in range(0, index, 2):
                        loss_tmp = loss_fn(score=output[i], feat=output[i + 1], target=target, target_cam=target_cam)
                        loss = loss + loss_tmp
            
            scaler.scale(loss).backward()
            # 🟡 **打印梯度范数，检查是否梯度爆炸**
            total_norm = 0.0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm(2).item()  # L2 范数
                    total_norm += grad_norm ** 2
            total_norm = total_norm ** 0.5  # 计算全局梯度范数
     

            # 🔴 **如果梯度过大，进行裁剪**
            if total_norm > 100:  # 你可以调整这个阈值
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)     
            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            if isinstance(output, list):
                acc = (output[0][0].max(1)[1] == target).float().mean()
            else:
                acc = (output[0].max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img['RGB'].shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                # print(scheduler._get_lr(epoch))
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                        .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = {'RGB': img['RGB'].to(device),
                                   'NI': img['NI'].to(device),
                                   'TI': img['TI'].to(device)}
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            feat = model(img, cam_label=camids, view_label=target_view)
                            if cfg.DATASETS.NAMES == "MSVR310":
                                evaluator.update((feat, vid, camid, target_view, _))
                            else:
                                evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            else:
                model.eval()
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = {'RGB': img['RGB'].to(device),
                               'NI': img['NI'].to(device),
                               'TI': img['TI'].to(device)}
                        camids = camids.to(device)
                        scenceids = target_view
                        target_view = target_view.to(device)
                        feat = model(img, cam_label=camids, view_label=target_view)
                        if cfg.DATASETS.NAMES == "MSVR310":
                            evaluator.update((feat, vid, camid, scenceids, _))
                        else:
                            evaluator.update((feat, vid, camid))
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                if mAP >= best_index['mAP']:
                    best_index['mAP'] = mAP
                    best_index['Rank-1'] = cmc[0]
                    best_index['Rank-5'] = cmc[4]
                    best_index['Rank-10'] = cmc[9]
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + 'best.pth'))
                logger.info("Best mAP: {:.1%}".format(best_index['mAP']))
                logger.info("Best Rank-1: {:.1%}".format(best_index['Rank-1']))
                logger.info("Best Rank-5: {:.1%}".format(best_index['Rank-5']))
                logger.info("Best Rank-10: {:.1%}".format(best_index['Rank-10']))
                torch.cuda.empty_cache()


def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("TOPReID.test")
    logger.info("Enter inferencing")

    if cfg.DATASETS.NAMES == "MSVR310":
        evaluator = R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
        evaluator.reset()
    else:
        evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
        evaluator.reset()
    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    all_feats = []          # 保存所有特征
    all_pids = []            # 保存所有行人ID
    all_camids = []          # 保存所有相机ID
    all_imgpaths = []  
    img_path_list = []
    dataset_root = cfg.DATASETS.ROOT_DIR
    logger.info(f"Dataset root directory: {dataset_root}")

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = {'RGB': img['RGB'].to(device),
                   'NI': img['NI'].to(device),
                   'TI': img['TI'].to(device)}
            camids = camids.to(device)
            scenceids = target_view
            target_view = target_view.to(device)
            feat = model(img, cam_label=camids, view_label=target_view)

            if cfg.DATASETS.NAMES == "MSVR310":
                evaluator.update((feat, pid, camid, scenceids, imgpath))
            else:
                evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

            all_feats.append(feat.cpu())
            all_pids.extend(pid.cpu().numpy() if torch.is_tensor(pid) else pid)
            all_camids.extend(camid.cpu().numpy() if torch.is_tensor(camid) else camid)
            all_imgpaths.extend(imgpath)
    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    
    try:
        # 拼接特征
        all_feats = torch.cat(all_feats, dim=0)
        
        # 区分query和gallery
        query_feats = all_feats[:num_query]
        gallery_feats = all_feats[num_query:]
        query_pids = np.array(all_pids[:num_query])
        gallery_pids = np.array(all_pids[num_query:])
        query_paths = all_imgpaths[:num_query]
        gallery_paths = all_imgpaths[num_query:]
        
        # 打印路径示例，确认是否正确
        logger.info(f"Query path example: {query_paths[0]}")
        logger.info(f"Gallery path example: {gallery_paths[0]}")
        
        # 计算余弦距离矩阵
        query_feats = F.normalize(query_feats, p=2, dim=1)
        gallery_feats = F.normalize(gallery_feats, p=2, dim=1)
        distmat = 1 - torch.mm(query_feats, gallery_feats.t())
        distmat = distmat.cpu().numpy()
        
        # 获取排序索引
        indices = np.argsort(distmat, axis=1)
        
        # 创建保存目录
        vis_dir = os.path.join(cfg.OUTPUT_DIR, 'rank_visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        # 设置可视化参数
        top_k = 10  # 显示前10个检索结果
        num_queries_to_vis = min(50, len(query_paths))
        
        logger.info(f"Generating rank visualizations for {num_queries_to_vis} queries...")
        
        for q_idx in range(num_queries_to_vis):
            q_path = query_paths[q_idx]
            q_pid = query_pids[q_idx]
            
            try:
                # 检查文件是否存在
                if not os.path.exists(q_path):
                    logger.warning(f"Query image not found: {q_path}")
                    continue
                    
                # 加载查询图像
                q_img = Image.open(q_path).convert('RGB')
                
                # 获取该查询的top_k索引
                top_k_indices = indices[q_idx, :top_k]
                
                # 计算画布尺寸
                img_width, img_height = q_img.size
                cols = min(5, top_k + 1)
                rows = (top_k + 1 + cols - 1) // cols
                
                # 创建大画布
                canvas_width = cols * img_width
                canvas_height = rows * img_height
                canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')
                
                # 放置查询图像（左上角）
                canvas.paste(q_img, (0, 0))
                
                # 在查询图像上绘制标签
                draw_q = ImageDraw.Draw(canvas)
                draw_q.rectangle([0, 0, img_width-1, 30], fill='blue')
                draw_q.text((10, 5), f"Query ID:{q_pid}", fill='white')
                
                # 放置检索结果
                for rank, g_idx in enumerate(top_k_indices):
                    g_path = gallery_paths[g_idx]
                    g_pid = gallery_pids[g_idx]
                    
                    # 检查图库图像是否存在
                    if not os.path.exists(g_path):
                        logger.warning(f"Gallery image not found: {g_path}")
                        continue
                    
                    # 计算位置
                    col = (rank + 1) % cols
                    row = (rank + 1) // cols
                    x = col * img_width
                    y = row * img_height
                    
                    # 加载图库图像
                    g_img = Image.open(g_path).convert('RGB')
                    canvas.paste(g_img, (x, y))
                    
                    # 绘制边框和标签
                    draw = ImageDraw.Draw(canvas)
                    border_color = 'green' if q_pid == g_pid else 'red'
                    draw.rectangle([x, y, x+img_width-1, y+img_height-1], outline=border_color, width=3)
                    
                    # 绘制排名标签
                    draw.rectangle([x, y, x+60, y+25], fill=border_color)
                    draw.text((x+5, y+5), f"R-{rank+1}", fill='white')
                    
                    # 绘制PID标签
                    draw.text((x+5, y+img_height-20), f"ID:{g_pid}", fill='white')
                
                # 保存拼接图
                save_path = os.path.join(vis_dir, f'query_{q_idx:04d}_pid{q_pid}_rank_vis.jpg')
                canvas.save(save_path, quality=95)
                
                if (q_idx + 1) % 10 == 0:
                    logger.info(f"Generated {q_idx + 1}/{num_queries_to_vis} visualizations")
                    
            except Exception as e:
                logger.warning(f"Failed to generate visualization for query {q_idx}: {e}")
                continue
        
        logger.info(f"All rank visualizations saved to {vis_dir}")
        
    except Exception as e:
        logger.error(f"Error generating rank visualizations: {e}")
        import traceback
        traceback.print_exc()
                  

    return cmc[0], cmc[4]
