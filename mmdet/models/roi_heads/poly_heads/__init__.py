from .fcn_poly_head import FCNPolyHead
from .naive_poly_head import NaivePolyHead
from .gcp_poly_head import GCPPolyHead
from .seg_poly_head import SegPolyHead
from .seg2ins_head import Seg2InsHead
from .cluster_seg2ins_head import ClusterSeg2InsHead

__all__ = ['FCNPolyHead', 'NaivePolyHead', 'GCPPolyHead', 'SegPolyHead', 'Seg2InsHead',
           'ClusterSeg2InsHead']
