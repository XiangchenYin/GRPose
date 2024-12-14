import os 
import json
import cv2
from concurrent.futures import ThreadPoolExecutor
NUM_THREADS = 72  # 可以根据你的系统调整线程数量

path = '/home/xcyin/Workspace/data/LAION-Human/Aesthetics_Human/mapping_file_training.json'

img_path = '/home/xcyin/Workspace/data/LAION-Human/Aesthetics_Human/images'

root = '/home/xcyin/Workspace/data/LAION-Human'

with open(path, "r",encoding='utf-8') as f:
    map_json = json.load(f)


def rm_imgs(idx, img, folder):
    name = str(idx).zfill(5)+'_' + img.split('.')[0]
    # print(name, name in list(map_json.keys()))

    if not name in list(map_json.keys()):
        x = os.path.join(folder, img)
        print(f'rm -rf {x}')
        os.system(f'rm -rf {x}')


# 删除元文件里没有的图片
length = len(map_json)
print(length)
for idx in range(79):
    folder = f"{img_path}/{str(idx).zfill(5)}"
    imgs = os.listdir(folder)
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
            futures = []
            for img in imgs:
                
                futures.append(executor.submit(rm_imgs, idx, img, folder))



# for name in list(map_json.keys()):
#     print(map_json[name])
#     info = map_json[name]
#     img_path = info['img_path']
#     # print(img_path)
#     # print(os.path.join(root, img_path))
#     print(os.path.exists( os.path.join(root, img_path) ))
#     # raise


# print(list(map_json.keys())[0:10])

# results = [i for i in list(map_json.keys()) if int(i.split('_')[0])<=79 and os.path.exists(os.path.join(img_path, i.split('_')[0], i.split('_')[1]+'.jpg')) ]
# print(len(results))
