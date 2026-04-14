import numpy as np

TILE_PIXELS = 32

def highlight_img(img, color=(255, 255, 255), alpha=0.30):
    """
    Add highlighting to an image
    """

    blend_img = img + alpha * (np.array(color, dtype=np.uint8) - img)
    blend_img = blend_img.clip(0, 255).astype(np.uint8)
    img[:, :, :] = blend_img


def highlight_view(map_imgs, history_masks, highlight_masks, width=8, height=8, tile_size = TILE_PIXELS):

    assert len(map_imgs) == len(history_masks) == len(highlight_masks), "输入列表长度不一致！"
    # import pdb; pdb.set_trace()
    # width = map_imgs[0].shape[0]
    # height = map_imgs[0].shape[1]
    updated_map_imgs = []
    updated_history_masks = []
    max_count = 200
    # 遍历每个环境
    for env_idx in range(len(map_imgs)):
        # 获取当前环境的数据
        map_img = map_imgs[env_idx]
        history_mask = history_masks[env_idx]
        highlight_mask = highlight_masks[env_idx]
        # 更新 history_mask
        # import pdb; pdb.set_trace()
        history_mask += highlight_mask.astype(int)
        
        # 遍历网格并高亮
        for j in range(0, height):
            for i in range(0, width):
                count = history_mask[i, j]*2  # 使用 history_mask 的计数值
                if count > 0:
                    ymin = j * tile_size
                    ymax = (j + 1) * tile_size
                    xmin = i * tile_size
                    xmax = (i + 1) * tile_size
                    # 根据计数值动态调整 alpha 或颜色强度
                    # alpha = min(0.3 + 0.1 * count, 1.0)  # 最大透明度不超过 1.0
                    alpha = 0.7
                    color = (0, max(0, 255 - int(255 * count / max_count)), 255)  # 颜色从淡蓝色到深蓝色渐变
                    highlight_img(map_img[ymin:ymax, xmin:xmax, :], color=color, alpha=alpha)
        
        # 保存更新后的 map_img 和 history_mask
        updated_map_imgs.append(map_img)
        updated_history_masks.append(history_mask)
    
    return updated_map_imgs, updated_history_masks