import os

path = '/mnt/pfs-mc0p4k/tts/team/digital_avatar_group/sunwenzhang/xcyin/LAION-Human/pose'

ids = 80

for i in range(ids):
        os.system(f'cd {path} && unzip {i:05d}.zip')