import libtmux
from tools.voxel_dict import load_occ_table
import os
import time
import click


@click.command()
@click.option('--start', default=0)
@click.option('--ntraj', default=1000)
def main(start, ntraj):
    s = libtmux.Server()
    env_id = start

    while(env_id < 10):
        try:
            t = load_occ_table(env_id)
            # print(t)
        except:
            pass
        print("env{}, size:{}".format(env_id, len(t)))
        if len(t) < ntraj and s.find_where({ "session_name": "linjun_{}".format(env_id) }) is None:
            os.system("bash run.sh {env_id}".format(env_id=env_id))
        elif len(t) >= ntraj:
            for i in range(10):
                if s.find_where({ "session_name": "linjun_{}".format(i) }) is not None:
                        s.find_where({ "session_name": "linjun_{}".format(i) }).kill_session()
            print("env {} finished.".format(env_id))
            env_id += 2
        else:
            time.sleep(60)

main()
     
