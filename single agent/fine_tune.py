import subprocess

cmds = ['python main.py --lr=5e-4 --clip_param=0.05  --seed=11'
        ]

for cmd in cmds:
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    for line in iter(p.stdout.readline, b''):
        msg = line.strip().decode('gbk')
        print(msg)