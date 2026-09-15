from dataclasses import dataclass
from typing import Any

import inspect
import numpy as np

@dataclass
class MonteCarloResults:
    iterations_irr: np.ndarray
    reference_irr: float
    irr_variations: np.ndarray
    iterations_eoh: np.ndarray
    reference_eoh: float
    eoh_variations: np.ndarray
    iterations_time_to_cod: np.ndarray
    reference_time_to_cod: float
    delays: np.ndarray
    iterations_capex: np.ndarray
    reference_capex: float
    capex_variations: np.ndarray
    iterations_bpb: np.ndarray
    reference_bpb: float
    delay_stiffness: float
    capex_stiffness: float
    eoh_stiffness: float
    linear_irr_variations: np.ndarray
    percentiles_points: np.ndarray
    irr_percentiles: np.ndarray
    linear_irr_percentiles: np.ndarray
    linear_irr_error: np.ndarray

    @classmethod
    def from_tuple(cls, result):
        if len(result) != 22:
            raise ValueError(f"my_irr_mc_function ha restituito {len(result)} elementi anziche 22")
        array_indexes = {0, 2, 3, 5, 6, 8, 9, 11, 12, 17, 18, 19, 20, 21}
        values = [np.asarray(v) if i in array_indexes else v for i, v in enumerate(result)]
        return cls(*values)

class IrrModelWrapper:
    """Adatta i valori della GUI alla funzione di calcolo."""

    def run(self, params: dict[str, Any]) -> MonteCarloResults:
        from core import my_irr_mc_function
        accepted = inspect.signature(my_irr_mc_function).parameters
        model_params = {
            key: value
            for key, value in params.items()
            if key in accepted
        }

        result = my_irr_mc_function(**model_params)
        return MonteCarloResults.from_tuple(result)

