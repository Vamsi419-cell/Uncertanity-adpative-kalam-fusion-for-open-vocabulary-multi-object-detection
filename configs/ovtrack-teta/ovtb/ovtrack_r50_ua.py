
_base_ = "./ovtrack_r50.py"


tracker = dict(

    type="UAKFOVTracker",

    # Original OVTracker settings
    init_score_thr=0.0001,
    obj_score_thr=0.0001,
    match_score_thr=0.5,

    memo_frames=10,

    momentum_embed=0.8,
    momentum_obj_score=0.5,

    match_metric="bisoftmax",
    match_with_cosine=True,
    contrastive_thr=0.5,




    uncertainty=dict(

        enabled=True,

        alpha=1.0,

        beta=1.0,

        ema_mu=0.8,
    ),




    adaptive_r=dict(

        enabled=True,

        mapping="linear",

        gamma=1.0,
    ),




    adaptive_w=dict(

        enabled=True,

        w_base=0.03,

        delta=0.10,
    ),
)


# 10K debugging subset
data = dict(

    test=dict(

        ann_file=(
            "data/ovtb/"
            "ovtb_10k_ann.json"
        ),
    )
)
