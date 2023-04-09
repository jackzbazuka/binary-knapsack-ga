import copy
import random
from dataclasses import dataclass
from json import load

import numpy as np
import numpy.ma as ma


@dataclass()
class BKS_Solver:
	def __init__(
		self,
		file_path: str,
		cx_pb: float = 0.8,
		mut_pb: float = 0.2,
		n_pop: int = 100,
		n_gen: int = 1000,
	):
		self.file_path = file_path
		self.dataset_name = file_path.split("/")[-1].split(".")[0]

		self.cx_pb = cx_pb
		self.mut_pb = mut_pb
		self.n_pop = n_pop
		self.n_gen = n_gen

		self.capacity: int
		self.weights: np.ndarray
		self.profits: np.ndarray
		self.optimal_solution: np.ndarray

		self.current_generation: np.ndarray  # Will contain the current generation of solutions

		self.load_data()
		self.print_dataset_stats()

	def load_data(self):
		with open(self.file_path) as f:
			data = load(f)

			self.capacity = data["capacity"]
			self.weights = np.array(data["weights"])
			self.profits = np.array(data["profits"])

			self.optimal_solution = np.array(data["optimal_solution"])

	def print_dataset_stats(self):
		print("--------------------DESCRIPTION--------------------")
		print(f"Dataset name: {self.dataset_name}")
		print(
			f"Capacity: {self.capacity} [{self.capacity/self.weights.sum() * 100}% of Total Available Weight]"
		)
		print(f"Total Available Weight: {self.weights.sum()}")
		print(f"Total Available Profit: {self.profits.sum()}")

	def init_population(self):
		self.current_generation = np.random.choice(
			[0, 1], (self.n_pop, len(self.weights)), p=[0.1, 0.9]
		)

	def fitness(self, ind: np.ndarray) -> float:
		p_sum = ma.array(self.profits, mask=ind).sum()
		w_sum = ma.array(self.weights, mask=ind).sum()

		try:
			return (1 / (self.capacity - w_sum)) * (p_sum / w_sum)
		except ZeroDivisionError:
			return 1 * (p_sum / w_sum)

	def selection(self):
		sorted_pop = sorted(
			self.current_generation, key=lambda x: self.fitness(x), reverse=True
		)

		p1, p2 = sorted_pop[:2]

		return p1, p2

	def crossover(self, p1: np.ndarray, p2: np.ndarray):
		if random.random() < self.cx_pb:
			p1c1, p1c2 = np.array_split(p1, 2, axis=0)
			p2c1, p2c2 = np.array_split(p2, 2, axis=0)

			c1 = np.concatenate((p1c1, p2c2))
			c2 = np.concatenate((p1c2, p2c1))

			return c1, c2
		else:
			return p1, p2

	def mutation(self, ind: np.ndarray):
		if random.random() < self.mut_pb:
			ind_cpy = copy.deepcopy(ind)
			temp_idx = random.randint(0, len(ind_cpy) - 1)
			ind_cpy[temp_idx] = 0 if temp_idx == 1 else 1
			return ind_cpy
		else:
			return ind

	def evolution(self):
		self.init_population()

		csv_data = []

		for i in range(self.n_gen):
			child_generation = []

			sorted_fit_inds = [
				(ind, self.fitness(ind)) for ind in self.current_generation
			]
			sorted_fit_inds.sort(key=lambda x: x[1], reverse=True)

			for p1, p2 in zip(sorted_fit_inds[::2], sorted_fit_inds[1::2]):
				p1_ind, p2_ind = p1[0], p2[0]
				c1, c2 = self.crossover(p1_ind, p2_ind)
				c1 = self.mutation(c1)
				c2 = self.mutation(c2)

				child_generation.append(c1)
				child_generation.append(c2)

			gen_mean = np.mean([e[1] for e in sorted_fit_inds])
			csv_data.append(
				[
					i,
					sorted_fit_inds[0][1],
					gen_mean,
					len(self.current_generation),
				]
			)

			print(
				f"""--------------------Genration {i}--------------------
			Best Fitness: {sorted_fit_inds[0][1]}
			Avg Fitness: {gen_mean}
			Population Size: {len(self.current_generation)}
			"""
			)

			self.current_generation = np.array(child_generation)

		sorted_fit_inds = [(ind, self.fitness(ind)) for ind in self.current_generation]
		sorted_fit_inds.sort(key=lambda x: x[1], reverse=True)

		gen_mean = np.mean([e[1] for e in sorted_fit_inds])

		# mask takes values which are false or 0
		p_sum = ma.array(self.profits, mask=sorted_fit_inds[0][0]).sum()
		w_sum = ma.array(self.weights, mask=sorted_fit_inds[0][0]).sum()

		print(
			f"""--------------------Evolution Complete--------------------
			Best Fitness: {sorted_fit_inds[0][1]}
			Avg Fitness: {gen_mean}
			Population Size: {len(self.current_generation)}
			Profit: {p_sum}
			Occupied Capacity: {(w_sum/self.capacity) * 100}% [{w_sum}/{self.capacity}]
			"""
		)

		with open(
			f"results/{self.dataset_name}_cx{self.cx_pb}_mx{self.mut_pb}_inst{len(self.weights)}_gen{self.n_gen}_pop{self.n_pop}.csv",
			"w",
		) as f:
			f.write("Generation,Best Fitness,Avg Fitness,Population Size\n")

			for row in csv_data:
				f.write(",".join([str(x) for x in row]) + "\n")
