from models import BKS_Solver

dataset_1000_name = "knapPI_3_10000_1000_1.json"
dataset_100_name = "knapPI_1_100_1000_1.json"

file_path = f"data/json/large_scale/{dataset_100_name}"

solver = BKS_Solver(file_path, cx_pb=0.8, mut_pb=0.1, n_pop=50, n_gen=100)
solver.evolution()
