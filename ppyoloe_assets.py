"""Pinned sources used to prepare the Objects365 detector."""

CHECKPOINT_URL = "https://bj.bcebos.com/v1/paddledet/models/pretrained/ppyoloe_crn_s_obj365_pretrained.pdparams"
# SHA-256 recorded from the official download, not an upstream-published digest.
CHECKPOINT_SHA256 = "a0b3edcf31d2c641c20ae4e137666662d498b48f43d09debeaea2dfb0c570ff7"
LABELS_URL = "https://bj.bcebos.com/v1/paddledet/data/objects365/objects365_detection_label_list.txt"
LABELS_SHA256 = "0dfa8029411a8a80df344d724719fad7cb6fb5421ef8ac4e95a09d5d8494c0bc"
PADDLEDETECTION_COMMIT = "b25522a0f4bde8c80603f3ba5e3472059972e3b5"
CONFIG_PATH = "configs/ppyoloe/objects365/ppyoloe_plus_crn_s_60e_objects365.yml"
