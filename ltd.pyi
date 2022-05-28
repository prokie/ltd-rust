from typing import List, Tuple, Union

def downsample(
    x: List[Union[float, int]], y: List[Union[float, int]], threshold: int
) -> Tuple[List[Union[float, int]], List[Union[float, int]]]:
    """
    Downsample using Large-Triangle-Dynamic.
    """
