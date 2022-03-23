import subprocess

cmds = ['python main.py --lr=5e-3 --clip_param=0.05 --ppo_update_time=8 --seed=11',
        # 'python main.py --lr=5e-4 --clip_param=0.05 --seed=17',
        # 'python main.py --lr=5e-4 --clip_param=0.1 --seed=33',
        # 'python main.py --lr=5e-4 --clip_param=0.01 --seed=233'
        ]

for cmd in cmds:
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    for line in iter(p.stdout.readline, b''):
        msg = line.strip().decode('gbk')
        print(msg)