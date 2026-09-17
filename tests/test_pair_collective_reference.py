from __future__ import annotations

import unittest

import numpy as np

from ftgspp.training.distributed_contract import DistributedContractError
from tests.reference.pair_collective import (
    SortedSparseGradient,
    choose_exact_bucket_codec,
    exact_sorted_sparse_mean,
    fixed_rank_dense_mean,
    scatter_sparse_rows,
    sorted_sparse_from_dense,
)


class PairCollectiveReferenceTests(unittest.TestCase):
    def test_one_view_per_rank_matches_the_serial_view_mean(self):
        for world_size in (2,):
            views = [
                np.full((4,), np.float32(1 + rank), dtype=np.float32)
                for rank in range(world_size)
            ]
            np.testing.assert_allclose(
                fixed_rank_dense_mean(views),
                np.full(
                    (4,), np.float32((world_size + 1) / 2), dtype=np.float32
                ),
                rtol=0.0,
                atol=1.0e-6,
            )

    def test_sparse_union_uses_full_world_size_divisor(self):
        dense = [
            np.asarray([[2.0, 4.0], [0.0, 0.0], [6.0, 8.0]], np.float32),
            np.asarray([[0.0, 0.0], [10.0, 12.0], [14.0, 16.0]], np.float32),
        ]
        touched = [
            np.asarray([True, False, True]),
            np.asarray([False, True, True]),
        ]
        records = [
            sorted_sparse_from_dense(value, mask)
            for value, mask in zip(dense, touched, strict=True)
        ]
        ids, values = exact_sorted_sparse_mean(
            records, row_count=3, record_width=2
        )
        np.testing.assert_array_equal(ids, np.asarray([0, 1, 2], np.uint32))
        np.testing.assert_array_equal(
            scatter_sparse_rows(ids, values, row_count=3),
            fixed_rank_dense_mean(dense),
        )

    def test_sparse_contract_rejects_unsorted_duplicate_ids(self):
        records = SortedSparseGradient(
            ids=np.asarray([2, 1, 1], np.uint32),
            values=np.ones((3, 2), np.float32),
        )
        with self.assertRaisesRegex(DistributedContractError, "strictly increasing"):
            records.validate(row_count=4)

    def test_sparse_codec_requires_exactness_and_five_percent_gain(self):
        self.assertEqual(
            choose_exact_bucket_codec(
                dense_seconds=1.0,
                sparse_seconds=0.9,
                sparse_exact=True,
            ),
            "exact_sorted_sparse",
        )
        self.assertEqual(
            choose_exact_bucket_codec(
                dense_seconds=1.0,
                sparse_seconds=0.9,
                sparse_exact=False,
            ),
            "dense",
        )
        self.assertEqual(
            choose_exact_bucket_codec(
                dense_seconds=1.0,
                sparse_seconds=0.96,
                sparse_exact=True,
            ),
            "dense",
        )


if __name__ == "__main__":
    unittest.main()
