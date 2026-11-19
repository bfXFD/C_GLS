import numpy as np

from scripts.fsm_bench_tracking_repair import choose, depth_score


def test_depth_score_uses_fixed_denominator_and_held_out_tiles():
    depth = np.ones((16, 16))
    points = np.array([[1., 1., 1.], [9., 1., 1.]])
    score = depth_score(points, np.eye(4), np.eye(3), depth)
    assert score == dict(agree=.5, contradiction=0., coverage=.5, score=.5)
    depth[:8, 8:] = 2.
    assert depth_score(points, np.eye(4), np.eye(3), depth) == score
    depth[:8, :8] = 2.
    assert depth_score(points, np.eye(4), np.eye(3), depth)["contradiction"] == .5


def test_choose_requires_validity_support_and_margin_not_truth():
    def hyp(i, score, agree, valid=True):
        return dict(index=i, depth=dict(score=score, agree=agree), valid=valid)
    baseline = hyp(0, .25, .26)
    assert choose([baseline, hyp(1, .26, .3), hyp(2, .8, .9, False)]) is baseline
    candidate = hyp(3, .30, .31)
    assert choose([baseline, candidate]) is candidate
