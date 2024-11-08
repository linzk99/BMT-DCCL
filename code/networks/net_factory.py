from networks.unet import *

def net_factory(net_type="unet", in_chns=1, class_num=3):
    if net_type == "unet":
        net = UNet(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_cct":
        net = UNet_CCT(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_cct_3h":
        net = UNet_CCT_3H(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_ds":
        net = UNet_DS(in_chns=in_chns, class_num=class_num).cuda()

    
   
    elif net_type == "UNet_contr":
        net = UNet_contr(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "UNet_BMT_DCCL":
        net = UNet_BMT_DCCL(in_chns=in_chns, class_num=class_num).cuda()
    
    
    else:
        net = None
    return net
