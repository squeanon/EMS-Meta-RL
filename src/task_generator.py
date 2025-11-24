from typing import Any, Dict, Optional, Type, List, Callable, Tuple
import random
import gymnasium

ListTask = List[Tuple[gymnasium.Env,Dict[str, Any]]]

# TODO: decaying revisits

class TaskGenerator:
    def __init__(
        self,
        tasks: Optional[ListTask] = None,
        task_callable: Optional[Callable] = None,
        task_callable_params: Optional[Dict[str, Any]] = None,
        revisit_ratio = 0.15,
        revisit_start = 0,
        sampling_method = "random",
        sampling_weights: Optional[List[float]] = None):
        """
        Initialize the TaskGenerator.

        :param tasks: List of predefined tasks (optional).
        :param task_callable: Callable that generates a task given a seed (optional).
        :param revisit_ratio: Proportion of tasks to revisit in each meta-iteration (between 0 and 1).
        :param revisit_start: Number of meta_steps before being to able to revisit tasks.
        :param sampling_method: Method for sampling tasks - "cyclic", "random", or "weighted".
        :param sampling_weights: Weights of tasks if using weighted sampling method.
        """

        assert tasks is not None or task_callable is not None, \
            "Either 'tasks' (list of tasks) or 'task_callable' (callable to generate tasks) must be provided."

        self.tasks = tasks  # List of predefined tasks, if any
        self.task_callable = task_callable  # Callable to generate tasks on the fly, returns tuple(env(gymenv),info(dict))
        self.task_callable_params = dict(task_callable_params or {})
        self.revisit_ratio = revisit_ratio  # Proportion of tasks to revisit
        self.revisit_start = revisit_start
        self.sampling_method = sampling_method  # Task sampling method
        self.sampling_weights = sampling_weights
        self.revisit_counter = 0

        self.selected_tasks = []  # Stores generated tasks for revisiting

    def reset_history(self):
        self.selected_tasks = []
        self.revisit_counter = 0

    def get_task(self, meta_step: int, seed: Optional[int] = None
             ) -> Tuple[gymnasium.Env, Dict[str, Any], Optional[int]]:
        """
            Generate or retrieve a task.

            Returns:
                (env, info, origin_meta_step)
                - env (gymnasium.Env): the environment/task instance
                - info (dict): task metadata
                - origin_meta_step (Optional[int]): meta-step of the first time this task spec appeared;
                None for static task lists.
        """
        # Option 1: Use a predefined list of tasks
        if self.tasks is not None:
            if len(self.tasks) == 0:
                raise ValueError("`tasks` must be non-empty.")

            if self.sampling_method == "cyclic":
                task, info = self.tasks[meta_step % len(self.tasks)]
            elif self.sampling_method == "random":
                task, info = random.choice(self.tasks)
            elif self.sampling_method == "weighted":
                if self.sampling_weights is None or len(self.sampling_weights) != len(self.tasks):
                    raise ValueError("`sampling_weights` must be provided and match the number of tasks.")
                task, info = random.choices(self.tasks, weights=self.sampling_weights, k=1)[0]
            else:
                raise ValueError(f"Unknown sampling_method: {self.sampling_method!r}")

            self.selected_tasks.append({'task': task, 'task_info': info, 'meta_step': [meta_step], 'seed': None})
            return task, info, None

        # Option 2: Dynamically generate tasks using the callable
        elif self.task_callable:
            can_revisit = (meta_step >= self.revisit_start) and bool(self.selected_tasks)

            if can_revisit and (random.random() < self.revisit_ratio)
                task_idx = self._select_task_index_for_revisit()
                record = self.selected_tasks[task_idx]
                task = record['task']
                info = record['task_info']
                record['meta_step'].append(meta_step)
                
                print(f"Revisiting task of meta_step: {self.selected_tasks[task_idx]['meta_step'][0]}")
                return task, info, record['meta_step'][0]

            # Generate a new seed and create a new task? do not revisit inentionally
            if seed is None
                seed = random.getrandbits(64)
            params = dict(self.task_callable_params or {})
            task, info = self.task_callable(random_seed=seed, **params)
            
            # Store the seed and task for future revisits
            self.selected_tasks.append({'task':task, 'seed':seed, 'task_info':info, 'meta_step': [meta_step]})
            return task, info, meta_step
        else:
            raise ValueError("Either 'tasks' or 'task_callable' must be provided.")
    
    def _select_task_index_for_revisit(self):
        """
        """
        if self.sampling_method == "cyclic":
            task_index = self.revisit_counter % len(self.selected_tasks)
            self.revisit_counter += 1
            return task_index
        
        elif self.sampling_method == "random":
            # Randomly select a seed from the previously generated seeds
            self.revisit_counter += 1
            return random.randint(0, len(self.selected_tasks)-1)
        
        elif self.sampling_method == "weighted":
            # return random.choices(self.tasks, weights=self.sampling_weights, k=1)[0]
            self.revisit_counter += 1
            raise NotImplementedError