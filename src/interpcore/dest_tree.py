import numpy as np
import logging
import time
import os
import concurrent.futures
from sklearn.neighbors import KDTree


from interpcore.config import InterpolationConfig, QUERY_TYPE
from interpcore.kernels import DEST_SRC_MAP, interpolate_block


class DestinationTree:
    def __init__(
        self,
        dest_coordinates: np.ndarray,
        src_coordinates: np.ndarray,
        dest_ids: np.ndarray,
        config: InterpolationConfig,
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
        config : InterpolationConfig
            Interpolation configuration parameters
        name : str, optional
            Name of the tree, by default None

        Attributes
        ----------
        tree : KDTree
            KDTree built on the destination coordinates for fast neighbor search
        name : str | None
            Name of the tree
        ids : np.ndarray
            IDs of the destination nodes or elements
        config : InterpolationConfig
            Interpolation configuration parameters
        src_coordinates : np.ndarray
            point coordinates array of source mesh points
        idx_query : np.ndarray
            Indices of the neighbors in the destination mesh for each source point,
            obtained from the query method specified in the config
        """
        # store data
        self.name = name
        self.ids = dest_ids
        self.config = config
        self.src_coordinates = src_coordinates
        self.dest_coordinates = dest_coordinates

        # Build KDTree and run query
        logging.info("Building KDTree...")
        tic = time.perf_counter()
        if DEST_SRC_MAP[config.kernel]:
            self.dest_tree = KDTree(dest_coordinates)
            self.src_tree = None
            # Distribute each source value to destination neighbours
            self.idx_query = self._run_query(
                dest_tree=self.dest_tree,
                src_coordinates=src_coordinates,
                param=config.param,
                method=config.method,
            )
        else:
            self.dest_tree = None
            self.src_tree = KDTree(src_coordinates)
            # For each destination point, get source neighbours
            self.idx_query = self._run_query(
                dest_tree=self.src_tree,
                src_coordinates=dest_coordinates,
                param=config.param,
                method=config.method,
            )
        toc = time.perf_counter()
        logging.info("Elapsed Time:")
        logging.info(toc - tic)

    @staticmethod
    def _run_query(
        method: QUERY_TYPE, dest_tree: KDTree, src_coordinates: np.ndarray, param
    ) -> np.ndarray:
        """run the requested query by config"""
        logging.info("Extracting nodes from KDTree...")
        tic = time.perf_counter()
        if method == QUERY_TYPE.RADIUS:
            idx = dest_tree.query_radius(src_coordinates, r=param)
        elif method == QUERY_TYPE.K:
            # check for int already done in config
            idx = dest_tree.query(src_coordinates, k=int(param))[1]
        else:
            raise ValueError(f"Unsupported query method: {method}")

        toc = time.perf_counter()
        logging.info("Elapsed Time:")
        logging.info(toc - tic)

        return idx

    def interpolate(
        self, vector_to_interpolate: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Perform the interpolation from EM to Mech nodes

        Parameters
        ----------
        vector_to_interpolate : np.ndarray
            whatever vector that needs to be interpolated onto the destination mesh

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Interpolated forces on mechanical nodes and unmapped forces
        """
        logging.info("Performing interpolation...")
        tic = time.perf_counter()

        if self.dest_tree is None:
            coords = self.dest_coordinates
        else:
            coords = self.src_coordinates

        if self.config.multithread:
            num_threads = os.cpu_count() or 1
            blocks = _get_blocks(coords, num_threads)

            results = []
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=num_threads
            ) as executor:
                futures = [
                    executor.submit(
                        interpolate_block,
                        self.src_coordinates,
                        self.dest_coordinates,
                        self.idx_query,
                        block,
                        vector_to_interpolate,
                        self.config,
                    )
                    for block in blocks
                ]
                for future in concurrent.futures.as_completed(futures):
                    results.append(future.result())

            interpolated = np.zeros(
                [self.dest_coordinates.shape[0], self.config.num_components]
            )
            unmapped = np.zeros([1, self.config.num_components])
            for interp, unm in results:
                interpolated += interp
                unmapped += unm
        else:
            interpolated, unmapped = interpolate_block(
                self.src_coordinates,
                self.dest_coordinates,
                self.idx_query,
                slice(0, coords.shape[0]),
                vector_to_interpolate,
                self.config,
            )

        toc = time.perf_counter()
        logging.info("Elapsed Time:")
        logging.info(toc - tic)

        return interpolated, unmapped


def _get_blocks(src_coord: np.ndarray, num_threads: int):
    block_size = int(np.ceil(src_coord.shape[0] / num_threads))
    blocks = [
        slice(
            i * block_size,
            min((i + 1) * block_size, src_coord.shape[0]),
        )
        for i in range(num_threads)
    ]
    return blocks
