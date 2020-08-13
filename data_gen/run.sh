#!/bin/bash
# run28.sh env_name num_traj

tmux new-session -s "linjun_$1" -d

for i in {0..32}; do
    tmux new-window -t "linjun_$1:$i" -n "th_$i"
    tmux select-window -t "linjun_$1:$i"
    tmux send-keys -t "linjun_$1:$i" "while [ true ]; do bash datagen_quadrotor_obs_batch1.sh $1;done; sleep 1" C-m
    sleep 2
done

# tmux split-window -v -p 60 -t $1

# tmux split-window -v -p 40 -t $1

# tmux split-window -v -p 20 -t $1

# echo "while [ $(ls -l $2/path | grep '^-' |wc -l) -lt 2000 ]; do python3 gen_path.py --env $2 ;done"

# tmux send-keys -t $1:0.0 "python3 gen_path.py --env $2" C-m
# sleep 2
# tmux send-keys -t $1:0.1 "python3 gen_path.py --env $2" C-m
# sleep 2
# tmux send-keys -t $1:0.2 "python3 gen_path.py --env $2" C-m
# sleep 2
# tmux send-keys -t $1:0.3 "python3 gen_path.py --env $2" C-m
# sleep 2

# tmux send-keys -t $1:0.0 "while [ \$(ls -l $2/path | grep '^-' |wc -l) -lt $3 ]; do python3 gen_path.py --env $2 --num $3;done; sleep 1" C-m
# sleep 2
# tmux send-keys -t $1:0.1 "while [ \$(ls -l $2/path | grep '^-' |wc -l) -lt $3 ]; do python3 gen_path.py --env $2 --num $3 ;done; sleep 1" C-m
# sleep 2
# tmux send-keys -t $1:0.2 "while [ \$(ls -l $2/path | grep '^-' |wc -l) -lt $3 ]; do python3 gen_path.py --env $2 --num $3 ;done; sleep 1" C-m
# sleep 2
# tmux send-keys -t $1:0.3 "while [ \$(ls -l $2/path | grep '^-' |wc -l) -lt $3 ]; do python3 gen_path.py --env $2 --num $3 ;done; sleep 1" C-m
# sleep 2


