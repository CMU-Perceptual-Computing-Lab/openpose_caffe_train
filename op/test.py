import opcaffe

params = {
    "batch_size" : 5,
    "stride": 8,
    "max_degree_rotations": "45.0",
    "crop_size_x": 368,
    "crop_size_y": 368,
    "center_perterb_max": 40.0,
    "center_swap_prob": 0.0,
    "scale_prob": 1.0,
    "scale_mins": "0.333333333333",
    "scale_maxs": "1.5",
    "target_dist": 0.600000023842,
    "number_max_occlusions": "2",
    "sigmas": "7.0",
    "models": "COCO_25B_23;COCO_25B_17;MPII_25B_16;PT_25B_15",
    "sources": "/media/raaj/Storage/openpose_train/dataset/lmdb_coco2017_foot;/media/raaj/Storage/openpose_train/dataset/lmdb_coco;/media/raaj/Storage/openpose_train/dataset/lmdb_mpii;/media/raaj/Storage/openpose_train/dataset/lmdb_pt2_train",
    "probabilities": "0.05;0.85;0.05;0.05",
    "source_background": "/media/raaj/Storage/openpose_train/dataset/lmdb_background",
    "normalization": 0,
    "add_distance": 0
}
myClass = opcaffe.OPCaffe(params)

while 1:
    batch = opcaffe.Batch()
    myClass.load(batch)
    #print batch.label.shape
    #print batch.data.shape
