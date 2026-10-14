
import torch
import torch.nn as nn
import torch.nn.functional as F

def gamma_curve(x, g):
    # Power Curve, Gamma Correction Curve
    y = torch.clamp(x, 1e-3, 1)
    y = y ** g
    return y

def s_curve(x, alpha=0.5, beta=1.0):
    # S-Curve, from "Personalization of image enhancement (CVPR 2010)"
    below_alpha = x <= alpha
    
    ratio_below_alpha = torch.clamp(x / alpha, 0, 1-1e-3)
    s_below_alpha = alpha - alpha * ((1 - ratio_below_alpha) ** beta)
    
    ratio_above_alpha = torch.clamp((x - alpha) / (1 - alpha), 0, 1-1e-3)
    s_above_alpha = alpha + (1 - alpha) * (ratio_above_alpha ** beta)
    
    return torch.where(below_alpha, s_below_alpha, s_above_alpha)


# inverse tone curve MSE loss
def img2mse_tone(x, y):
    eta=1e-4
    x = torch.clip(x, min = eta, max = 1-eta)
    # the inverse tone curve, pls refer to paper (Eq.13): 
    # "https://openaccess.thecvf.com/content/ICCV2021/papers/Cui_Multitask_AET_With_Orthogonal_Tangent_Regularity_for_Dark_Object_Detection_ICCV_2021_paper.pdf"
    f=lambda x: 0.5 - torch.sin(torch.asin(1.0 - 2.0 * x) / 3.0)
    return torch.mean((f(x) - f(y)) ** 2)

def curve_loss(x, y, z):
    dim = x.shape[-1]
    coor = torch.linspace(0, 1, dim).unsqueeze(0).to(x.device)
    der_x = (x[:, 1:] - x[:, :-1]) / (coor[:, 1:] - coor[:, :-1])
    der_y = (y[:, 1:] - y[:, :-1]) / (coor[:, 1:] - coor[:, :-1])
    der_z = (z[:, 1:] - z[:, :-1]) / (coor[:, 1:] - coor[:, :-1])
    # consine similarity
    loss = (1 - torch.mean(F.cosine_similarity(der_x, der_y))) + \
       (1 - torch.mean(F.cosine_similarity(der_y, der_z))) + \
       (1- torch.mean(F.cosine_similarity(der_x, der_z))) 
    return loss
    
class HistogramPriorLoss(nn.Module):
    def __init__(self, lambda_smooth=0.1):
        
        super(HistogramPriorLoss, self).__init__()
        self.lambda_smooth = lambda_smooth
        #self.lambda_reg = lambda_reg

    def compute_histogram_equalization(self, input):
        # Resize Images
        input = torch.mean(nn.functional.interpolate(input.permute(0,3,1,2),scale_factor=0.25),dim=1)
        flat = input.flatten()
        hist = torch.histc(flat, bins=255, min=0.0, max=1.0)
        cdf = torch.cumsum(hist, dim=0)
        cdf = cdf / cdf[-1]  
        return cdf.unsqueeze(0)  

    def forward(self, output, input, psedo_curve, step):
        
        hist_eq_prior = self.compute_histogram_equalization(input)
        
        curve_loss = torch.mean((output - hist_eq_prior) ** 2)
        
        psedo_curve_loss = torch.mean((psedo_curve - output) ** 2) + 0.01 * torch.mean((psedo_curve - hist_eq_prior) ** 2)

        smooth_loss = torch.mean((output[:, 1:] - output[:, :-1]) ** 2)

        total_loss = curve_loss + self.lambda_smooth * smooth_loss + 0.5 * psedo_curve_loss

        if step >= 3000:
            total_loss = 0.5 * curve_loss + self.lambda_smooth * smooth_loss + 0.5 * psedo_curve_loss

        return total_loss

class AdaptiveCurveLoss(nn.Module):
    def __init__(self, alpha=0.2, beta=0.6, low_thresh=0.2, high_thresh=0.6, lambda1=1.0, lambda2=1.0, lambda3=0.1):
        
        super(AdaptiveCurveLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.low_thresh = low_thresh
        self.high_thresh = high_thresh
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3

    def forward(self, output):
        
        low_mask = (output < self.low_thresh).float()
        low_light_loss = torch.mean(low_mask * torch.abs(output - self.alpha))
        
        high_mask = (output > self.high_thresh).float()
        high_light_loss = torch.mean(high_mask * torch.abs(output - self.beta))
        
        smooth_loss = torch.mean((output[:, 1:] - output[:, :-1]) ** 2)

        total_loss = (
            self.lambda1 * low_light_loss +
            self.lambda2 * high_light_loss +
            self.lambda3 * smooth_loss
        )

        return total_loss

# Colour Constancy Loss
class L_color(nn.Module):

    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x):

        b,c,h,w = x.shape

        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr-mg,2)
        Drb = torch.pow(mr-mb,2)
        Dgb = torch.pow(mb-mg,2)
        k = torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2),0.5)

        return k

# Exposure Loss, control the generated image exposure
class L_exp(nn.Module):

    def __init__(self,patch_size,mean_val):
        super(L_exp, self).__init__()
        
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val

    def forward(self, x):

        b,c,h,w = x.shape
        x = torch.mean(x,1,keepdim=True)
        mean = self.pool(x)

        d = torch.mean(torch.pow(mean- torch.FloatTensor([self.mean_val]).to(x.device),2))
        return d

class L_spa(nn.Module):
    
    def __init__(self):
        super(L_spa, self).__init__()
        
        kernel_left = torch.FloatTensor( [[0,0,0],[-1,1,0],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor( [[0,0,0],[0,1,-1],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor( [[0,-1,0],[0,1, 0 ],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor( [[0,0,0],[0,1, 0],[0,-1,0]]).cuda().unsqueeze(0).unsqueeze(0)
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4)

    def forward(self, org , enhance, contrast=8):
        b,c,h,w = org.shape
        
        org_mean = torch.mean(org,1,keepdim=True)
        enhance_mean = torch.mean(enhance,1,keepdim=True)

        org_pool =  self.pool(org_mean)			
        enhance_pool = self.pool(enhance_mean)	
        
        D_org_letf = F.conv2d(org_pool , self.weight_left.to(org_pool.device), padding=1)
        D_org_right = F.conv2d(org_pool , self.weight_right.to(org_pool.device), padding=1)
        D_org_up = F.conv2d(org_pool , self.weight_up.to(org_pool.device), padding=1)
        D_org_down = F.conv2d(org_pool , self.weight_down.to(org_pool.device), padding=1)

        D_enhance_letf = F.conv2d(enhance_pool , self.weight_left.to(org_pool.device), padding=1)
        D_enhance_right = F.conv2d(enhance_pool , self.weight_right.to(org_pool.device), padding=1)
        D_enhance_up = F.conv2d(enhance_pool , self.weight_up.to(org_pool.device), padding=1)
        D_enhance_down = F.conv2d(enhance_pool , self.weight_down.to(org_pool.device), padding=1)

        D_left = torch.pow(D_org_letf * contrast - D_enhance_letf,2)
        D_right = torch.pow(D_org_right * contrast - D_enhance_right,2)
        D_up = torch.pow(D_org_up * contrast - D_enhance_up,2)
        D_down = torch.pow(D_org_down * contrast - D_enhance_down,2)
        E = (D_left + D_right + D_up +D_down)
        
        
        return torch.mean(E)

def loss_structure(struct_confidence, enhanced, original, weight=1.0):
    """结构保真损失：约束结构区域的几何一致性

    Args:
        struct_confidence: 结构置信度图（连续值）
        enhanced: 增强后的图像
        original: 原始图像
        weight: 损失权重

    Returns:
        结构保真损失值
    """
    if struct_confidence.shape[1:3] != enhanced.shape[1:3]:
        struct_confidence = F.interpolate(struct_confidence.permute(0, 3, 1, 2), size=enhanced.shape[1:3], mode='bilinear', align_corners=False).permute(0, 2, 3, 1)

    if struct_confidence.shape != enhanced.shape:
        if struct_confidence.shape[-1] != enhanced.shape[-1]:
            struct_confidence = struct_confidence.expand_as(enhanced)

    struct_region_enh = enhanced * struct_confidence
    struct_region_gt = original * struct_confidence

    loss = F.l1_loss(struct_region_enh, struct_region_gt)

    return weight * loss

def loss_noise(flat_mask, dark_mask, enhanced, weight=1.0, noise_threshold=0.01, min_pixels=100):
    """噪声感知损失：约束平坦暗区的噪声方差
    
    Args:
        flat_mask: 平坦区域掩码
        dark_mask: 暗区域掩码
        enhanced: 增强后的图像
        weight: 损失权重
        noise_threshold: 噪声方差阈值
        min_pixels: 最小像素数阈值
    
    Returns:
        噪声感知损失值
    """
    # 确保掩码和图像形状匹配
    if flat_mask.shape != dark_mask.shape:
        # 调整掩码尺寸以匹配
        dark_mask = F.interpolate(dark_mask.permute(0, 3, 1, 2), size=flat_mask.shape[1:3], mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
    
    if flat_mask.shape != enhanced.shape:
        # 调整掩码尺寸以匹配增强图像
        flat_mask = F.interpolate(flat_mask.permute(0, 3, 1, 2), size=enhanced.shape[1:3], mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
        dark_mask = F.interpolate(dark_mask.permute(0, 3, 1, 2), size=enhanced.shape[1:3], mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
    
    # 确保掩码的通道数与增强图像的通道数匹配
    if flat_mask.shape[-1] != enhanced.shape[-1]:
        flat_mask = flat_mask.expand_as(enhanced)
        dark_mask = dark_mask.expand_as(enhanced)
    
    # 计算平坦暗区域掩码
    flat_dark_mask = flat_mask * dark_mask
    
    # 检查掩码中是否有足够的像素
    if torch.sum(flat_dark_mask) < min_pixels:
        return 0.0
    
    # 提取平坦暗区域的像素值
    flat_dark_region = enhanced[flat_dark_mask > 0]
    
    # 计算方差
    var = torch.var(flat_dark_region)
    
    # 计算噪声损失
    loss = torch.max(var - noise_threshold, torch.tensor(0.0, device=enhanced.device))
    
    return weight * loss

def loss_multi_view(views, weight=1.0):
    """多视图一致性损失：约束不同视角下的颜色一致性
    
    Args:
        views: 不同视角的增强图像列表
        weight: 损失权重
    
    Returns:
        多视图一致性损失值
    """
    if len(views) < 2:
        return 0.0
    
    loss = 0.0
    num_views = len(views)
    
    # 计算所有视图对之间的一致性损失
    for i in range(num_views):
        for j in range(i + 1, num_views):
            # 确保两个视图的形状匹配
            view1 = views[i]
            view2 = views[j]
            
            # 如果形状不匹配，调整尺寸
            if view1.shape != view2.shape:
                # 调整到较小的尺寸
                min_height = min(view1.shape[1], view2.shape[1])
                min_width = min(view1.shape[2], view2.shape[2])
                view1 = F.interpolate(view1.permute(0, 3, 1, 2), size=(min_height, min_width), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
                view2 = F.interpolate(view2.permute(0, 3, 1, 2), size=(min_height, min_width), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
            
            # 计算L1损失
            loss += F.l1_loss(view1, view2)
    
    # 平均损失
    num_pairs = num_views * (num_views - 1) / 2
    if num_pairs > 0:
        loss = loss / num_pairs
    
    return weight * loss

if __name__ == '__main__':
    
    x_in_low = torch.rand(1,3,399,499)  # Pred normal-light
    x_in_enh = torch.rand(1,3,399,499)  # Pred normal-light
    x_gt = torch.rand(1,3,399,499)  # GT low-light

    curve_1 = torch.linspace(0, 1, 255).unsqueeze(0)
    curve_2 = gamma_curve(curve_1, 1.0)
    curve_3 = s_curve(curve_2, alpha=1.0, beta=1.0)
    

    
    

    
    
    
