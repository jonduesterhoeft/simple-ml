import random
from typing import TypeVar, Callable, List

T = TypeVar('T')
Stat = TypeVar('Stat')

def bootstrap_sample(data: List[T]) -> List[T]:
    return[random.choice(data) for _ in data]
        

def bootstrap_statistic(data: List[x],
                        stat_func: Callable[[List[T]], Stat],
                        num_samples: int) -> List[Stat]:
    return[stat_func(bootstrap_sample(data) for _ in range(num_samples))]