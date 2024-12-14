from pandas import read_parquet
import os
import requests
from concurrent.futures import ThreadPoolExecutor

BASE_DIR = "/home/xcyin/Workspace/data/LAION-Human/Aesthetics_Human"
NUM_THREADS = 72  # 可以根据你的系统调整线程数量
TIMEOUT = 5  # 请求超时时间

def download_image(url, key, idx):
    image_path = os.path.join(BASE_DIR, "images", str(idx).zfill(5), key) + '.jpg'

    if not os.path.exists(os.path.dirname(image_path)):
        os.makedirs(os.path.dirname(image_path))

    if os.path.exists(image_path):
        print(f"Image {key}.jpg already exists!")
        return

    try:
        r = requests.get(url, timeout=TIMEOUT)

        if r.status_code != 200:
            print(f"Error! Unable to download image {key}.jpg")
            return

        with open(image_path, 'wb') as f:
            f.write(r.content)

        print(f"Successfully downloaded image {key}.jpg")
    except Exception as e:
        print(f"Error! Unable to download image {key}.jpg: {e}")

for idx in range(74, 80): # 287
    data = read_parquet(f"{BASE_DIR}/images/{str(idx).zfill(5)}.parquet")

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = []
        for i in range(len(data)):
            url = data["url"][i]
            key = data["key"][i]
            futures.append(executor.submit(download_image, url, key, idx))
        
        # 可选：等待所有任务完成
        for future in futures:
            try:
                future.result()
            except Exception as e:
                print(f"Error! {e}")
  