import os
import config
import subprocess

def run():

    session = "a3c"

    cmds = [
        "tmux kill-session -t {}".format(session),
    ]
    '''excute cmds'''
    os.system("\n".join(cmds))

    subprocess.call(["rm", "-r", 'temp'])


if __name__ == "__main__":
    run()
