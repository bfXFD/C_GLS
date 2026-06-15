data_config = dict(
    basedir="/home/ras/xbf/SplaTAM/data/Replica",
    gradslam_data_cfg="./configs/data/replica.yaml",
    sequence="room0",
    desired_image_height=680,
    desired_image_width=1200,
    start=0,
    end=-1,
    stride=1,
    depth_min=0.0,
    depth_max=10.0,
    max_points=10000,
)

config = dict(
    seed=0,
    primary_device="cuda:0",
    data=data_config,
    dataset=data_config,
    registration=dict(
        backend="numpy_point_to_point",
        voxel_size=None,
        max_correspondence_distance=0.07,
        max_iterations=30,
        normal_radius=0.10,
        normal_max_nn=30,
        init_mode="previous_estimate",
        fallback_on_divergence=True,
        max_pose_translation_norm=100.0,
    ),
    eval=dict(
        align_ate=True,
        rpe_delta=1,
        save_tum_format=True,
    ),
)
