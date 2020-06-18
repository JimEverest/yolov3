def build_targets(model, targets):
    # targets = [image, class, x, y, w, h]

    nt = targets.shape[0]
    tcls, tbox, indices, av = [], [], [], []
    multi_gpu = type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)
    reject, use_all_anchors = True, True
    for i in model.yolo_layers:
        # get number of grid points and anchor vec for this yolo layer
        # 获取对应yolo层的grid尺寸和anchor大小

        if multi_gpu:
            ng, anchor_vec = model.module.module_list[i].ng, model.module.module_list[i].anchor_vec
        else:
            ng, anchor_vec = model.module_list[i].ng, model.module_list[i].anchor_vec
            #【13,13】 ， 【[3.62500, 2.81250], [4.87500, 6.18750], [11.65625, 10.18750]】   
            #              ([116, 90],         [156, 198],         [373, 326] / 32)
        # iou of targets-anchors
        t, a = targets, []
        gwh = t[:, 4:6] * ng  #target's wh * 13     t[:, 4:6] : 原来为对于整张图的normalized(/416)，占据整张图比例  --->  乘回13， 现在数字表示占据多少grids。
        if nt:
            iou = wh_iou(anchor_vec, gwh)  # iou(3,n) = wh_iou(anchor_vec(3,2), gwh(n,2))
                                           # 代表 3个anchor与每个ground truth的iou
                                           # wh_iou 只考虑wh， 不计算xy偏差
            if use_all_anchors:
                na = anchor_vec.shape[0]  # number of anchors
                a = torch.arange(na).view((-1, 1)).repeat([1, nt]).view(-1)   #nt: targets count
                t = targets.repeat([na, 1])                                   
                gwh = gwh.repeat([na, 1])
            else:  # use best anchor only
                iou, a = iou.max(0)  # best iou and anchor

            # reject anchors below iou_thres (OPTIONAL, increases P, lowers R)
            if reject:
                j = iou.view(-1) > model.hyp['iou_t']  # iou threshold hyperparameter
                t, a, gwh = t[j], a[j], gwh[j]  #24--->19

        # Indices
        b, c = t[:, :2].long().t()  # target image, class   long--->长整形  t() ---> 转置
        gxy = t[:, 2:4] * ng  # grid x, y
        gi, gj = gxy.long().t()  # grid x, y indices # 注意这里通过long将其转化为整形，代表格子的左上角
        indices.append((b, a, gj, gi))
        # indice结构体保存内容为：
        '''
        b: 一个batch中的角标
        a: 代表所选中的正样本的anchor的下角标
        gj, gi: 代表target所属grid的左上角坐标（编号）
        '''

        # Box
        gxy -= gxy.floor()  # xy
        tbox.append(torch.cat((gxy, gwh), 1))  # xywh (grids)
        av.append(anchor_vec[a])  # anchor vec

        # Class
        tcls.append(c)
        if c.shape[0]:  # if any targets
            assert c.max() < model.nc, 'Model accepts %g classes labeled from 0-%g, however you labelled a class %g. ' \
                                       'See https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data' % (
                                           model.nc, model.nc - 1, c.max())

    return tcls, tbox, indices, av
