# -*- coding: utf-8 -*-

import os

import pytest


def test_find_image_synthetic(tmp_path):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")

    # create synthetic screenshot
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    # white rectangle template location
    img[80:120, 140:200] = (255, 255, 255)

    shot = tmp_path / "shot.png"
    tpl = tmp_path / "tpl.png"
    cv2.imwrite(str(shot), img)
    cv2.imwrite(str(tpl), img[80:120, 140:200])

    from hmdriver2._vision import find_image

    r = find_image(str(shot), str(tpl), threshold=0.99, grayscale=True)
    assert r is not None
    assert abs(r.x - 140) <= 1
    assert abs(r.y - 80) <= 1

