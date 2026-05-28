import numpy as np
import logging
import time
from sklearn.neighbors import KDTree

from abc import ABC, abstractmethod


class DestinationTree(ABC):
    def __init__(
        self,
        dest_coordinates: np.ndarray,
        src_coordinates: np.ndarray,
        dest_ids: np.ndarray,
        name: str | None = None,
    ):
        """Intialize the KDTree for fast search in the mechanical nodes

        Parameters
        ----------
        dest_coordinates : np.ndarray
            point coordinates array of destination mesh points
        src_coordinates : np.ndarray
            point coordinates array of source mesh points
        dest_ids : np.ndarray
            IDs of the destination nodes or elements
        name : str, optional
            Name of the tree, by default None
        """
        logging.info("Building KDTree...")
        tic = time.perf_counter()
        self.tree = KDTree(dest_coordinates)
        toc = time.perf_counter()
        logging.info("Elapsed Time:")
        logging.info(toc - tic)
        self.name = name
        self.ids = dest_ids

    def _run_query(self):
        """run the requested query by config"""
        logging.info("Extracting nodes from KDTree...")
        tic = time.perf_counter()
        if self.config.method == QUERY_TYPE.RADIUS:
            idx = self.tree.query_radius(self.X_EM, r=self.config.param)
        elif self.config.method == QUERY_TYPE.K:
            # check for int already done in config
            idx = self.tree.query(self.X_EM, k=self.config.param)[1]

        toc = time.perf_counter()
        logging.info("Elapsed Time:")
        logging.info(toc - tic)

        return idx

    def interpolate(self, F_EM: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Perform the interpolation from EM to Mech nodes

        Parameters
        ----------
        F_EM : np.ndarray
            Nodal force array from EM analysis

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Interpolated forces on mechanical nodes and unmapped forces
        """
        logging.info("Performing interpolation...")
        tic = time.perf_counter()

        if self.config.multithread:
            num_threads = os.cpu_count() or 1
            block_size = int(np.ceil(self.X_EM.shape[0] / num_threads))
            blocks = [
                slice(i * block_size, min((i + 1) * block_size, self.X_EM.shape[0]))
                for i in range(num_threads)
            ]

            results = []
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=num_threads
            ) as executor:
                futures = [
                    executor.submit(
                        _interpolate_block,
                        self.X_EM,
                        self.X_Mech,
                        self.idx,
                        block,
                        F_EM,
                        self.config,
                    )
                    for block in blocks
                ]
                for future in concurrent.futures.as_completed(futures):
                    results.append(future.result())

            interpolated = np.zeros([self.X_Mech.shape[0], 3])
            unmapped = np.zeros([1, 3])
            for interp, unm in results:
                interpolated += interp
                unmapped += unm
        else:
            interpolated, unmapped = _interpolate_block(
                self.X_EM,
                self.X_Mech,
                self.idx,
                slice(0, self.X_EM.shape[0]),
                F_EM,
                self.config,
            )

        toc = time.perf_counter()
        logging.info("Elapsed Time:")
        logging.info(toc - tic)

        return interpolated, unmapped
