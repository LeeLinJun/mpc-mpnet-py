#! /bin/bash
source activate linjun
python exported/export_mpnet.py
python exported/export_cost_to_go.py
python exported/export_cost_so_far.py
python exported/export_cost.py
