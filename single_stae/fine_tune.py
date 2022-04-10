import subprocess

cmds = [
    'python main.py --ppo_update_time=4 --clip_param=0.05 --critic_lr=5e-4 --seed=333 --max_episode=10000'

# 'python main.py --ppo_update_time=6 --clip_param=0.05 --batch_size=32 --seed=632 --max_episode=20000'
]

for cmd in cmds:
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    for line in iter(p.stdout.readline, b''):
        msg = line.strip().decode('gbk')
        print(msg)

print('all done')