"""Pro try-on mask helpers for the VASTRA backend.

`get_mask_location` is vendored from OOTDiffusion (CC BY-NC-SA 4.0):
https://github.com/levihsu/OOTDiffusion/blob/main/run/utils_ootd.py
Our change: a `fit` parameter ('slim' / 'regular' / 'loose') that grows or
shrinks the try-on repaint mask, so the AI paints a tighter or roomier
garment. Same control principle as the "mask adjust" step in FitDiT/Leffa.

`restore_skin` + `composite_final` are our body-preservation guarantee:
after generation we paste the ORIGINAL photo pixels back everywhere the new
garment does not cover (face, hair, arms, legs, background...).

NOTE: the Kaggle notebook embeds a copy of this file (Kaggle cannot import
from here). Keep both in sync — test with:  python3 ootd_mask_fit.py
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw

label_map = {
    "background": 0,
    "hat": 1,
    "hair": 2,
    "sunglasses": 3,
    "upper_clothes": 4,
    "skirt": 5,
    "pants": 6,
    "dress": 7,
    "belt": 8,
    "left_shoe": 9,
    "right_shoe": 10,
    "head": 11,
    "left_leg": 12,
    "right_leg": 13,
    "left_arm": 14,
    "right_arm": 15,
    "bag": 16,
    "scarf": 17,
}


def extend_arm_mask(wrist, elbow, scale):
    wrist = elbow + scale * (wrist - elbow)
    return wrist


def hole_fill(img):
    img = np.pad(img[1:-1, 1:-1], pad_width=1, mode='constant', constant_values=0)
    img_copy = img.copy()
    mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), dtype=np.uint8)
    cv2.floodFill(img, mask, (0, 0), 255)
    img_inverse = cv2.bitwise_not(img)
    dst = cv2.bitwise_or(img_copy, img_inverse)
    return dst


def refine_mask(mask):
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8),
                                           cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
    area = []
    for j in range(len(contours)):
        a_d = cv2.contourArea(contours[j], True)
        area.append(abs(a_d))
    refine_mask = np.zeros_like(mask).astype(np.uint8)
    if len(area) != 0:
        i = area.index(max(area))
        cv2.drawContours(refine_mask, contours, i, color=255, thickness=-1)
    return refine_mask


# Fit control: how many dilation passes grow the repaint region.
FIT_DILATE = {"slim": 2, "regular": 5, "loose": 9}
FIT_ARM_ADJ = {"slim": -8, "regular": 0, "loose": 10}
FIT_BOTTOM_MARGIN = {"slim": 8, "regular": 20, "loose": 32}


def get_mask_location(model_type, category, model_parse: Image.Image, keypoint: dict,
                      fit="regular", width=384, height=512):
    im_parse = model_parse.resize((width, height), Image.NEAREST)
    parse_array = np.array(im_parse)

    if model_type == 'hd':
        arm_width = 60
    elif model_type == 'dc':
        arm_width = 45
    else:
        raise ValueError("model_type must be 'hd' or 'dc'!")
    arm_width += FIT_ARM_ADJ.get(fit, 0)

    parse_head = (parse_array == 1).astype(np.float32) + \
                 (parse_array == 3).astype(np.float32) + \
                 (parse_array == 11).astype(np.float32)

    parser_mask_fixed = (parse_array == label_map["left_shoe"]).astype(np.float32) + \
                        (parse_array == label_map["right_shoe"]).astype(np.float32) + \
                        (parse_array == label_map["hat"]).astype(np.float32) + \
                        (parse_array == label_map["sunglasses"]).astype(np.float32) + \
                        (parse_array == label_map["bag"]).astype(np.float32)

    parser_mask_changeable = (parse_array == label_map["background"]).astype(np.float32)

    arms_left = (parse_array == 14).astype(np.float32)
    arms_right = (parse_array == 15).astype(np.float32)
    arms = arms_left + arms_right

    if category == 'dresses':
        parse_mask = (parse_array == 7).astype(np.float32) + \
                     (parse_array == 4).astype(np.float32) + \
                     (parse_array == 5).astype(np.float32) + \
                     (parse_array == 6).astype(np.float32)
        parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))
    elif category == 'upper_body':
        parse_mask = (parse_array == 4).astype(np.float32) + (parse_array == 7).astype(np.float32)
        parser_mask_fixed_lower_cloth = (parse_array == label_map["skirt"]).astype(np.float32) + \
                                        (parse_array == label_map["pants"]).astype(np.float32)
        parser_mask_fixed += parser_mask_fixed_lower_cloth
        parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))
    elif category == 'lower_body':
        parse_mask = (parse_array == 6).astype(np.float32) + \
                     (parse_array == 12).astype(np.float32) + \
                     (parse_array == 13).astype(np.float32) + \
                     (parse_array == 5).astype(np.float32)
        parser_mask_fixed += (parse_array == label_map["upper_clothes"]).astype(np.float32) + \
                             (parse_array == 14).astype(np.float32) + \
                             (parse_array == 15).astype(np.float32)
        parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))
    else:
        raise NotImplementedError

    # Load pose points
    pose_data = keypoint["pose_keypoints_2d"]
    pose_data = np.array(pose_data)
    pose_data = pose_data.reshape((-1, 2))

    im_arms_left = Image.new('L', (width, height))
    im_arms_right = Image.new('L', (width, height))
    arms_draw_left = ImageDraw.Draw(im_arms_left)
    arms_draw_right = ImageDraw.Draw(im_arms_right)
    if category == 'dresses' or category == 'upper_body':
        shoulder_right = np.multiply(tuple(pose_data[2][:2]), height / 512.0)
        shoulder_left = np.multiply(tuple(pose_data[5][:2]), height / 512.0)
        elbow_right = np.multiply(tuple(pose_data[3][:2]), height / 512.0)
        elbow_left = np.multiply(tuple(pose_data[6][:2]), height / 512.0)
        wrist_right = np.multiply(tuple(pose_data[4][:2]), height / 512.0)
        wrist_left = np.multiply(tuple(pose_data[7][:2]), height / 512.0)
        ARM_LINE_WIDTH = int(arm_width / 512 * height)
        size_left = [shoulder_left[0] - ARM_LINE_WIDTH // 2, shoulder_left[1] - ARM_LINE_WIDTH // 2,
                     shoulder_left[0] + ARM_LINE_WIDTH // 2, shoulder_left[1] + ARM_LINE_WIDTH // 2]
        size_right = [shoulder_right[0] - ARM_LINE_WIDTH // 2, shoulder_right[1] - ARM_LINE_WIDTH // 2,
                      shoulder_right[0] + ARM_LINE_WIDTH // 2, shoulder_right[1] + ARM_LINE_WIDTH // 2]

        if wrist_right[0] <= 1. and wrist_right[1] <= 1.:
            im_arms_right = arms_right
        else:
            wrist_right = extend_arm_mask(wrist_right, elbow_right, 1.2)
            arms_draw_right.line(np.concatenate((shoulder_right, elbow_right, wrist_right)).astype(np.uint16).tolist(),
                                 'white', ARM_LINE_WIDTH, 'curve')
            arms_draw_right.arc(size_right, 0, 360, 'white', ARM_LINE_WIDTH // 2)

        if wrist_left[0] <= 1. and wrist_left[1] <= 1.:
            im_arms_left = arms_left
        else:
            wrist_left = extend_arm_mask(wrist_left, elbow_left, 1.2)
            arms_draw_left.line(np.concatenate((wrist_left, elbow_left, shoulder_left)).astype(np.uint16).tolist(),
                                'white', ARM_LINE_WIDTH, 'curve')
            arms_draw_left.arc(size_left, 0, 360, 'white', ARM_LINE_WIDTH // 2)

        hands_left = np.logical_and(np.logical_not(im_arms_left), arms_left)
        hands_right = np.logical_and(np.logical_not(im_arms_right), arms_right)
        parser_mask_fixed += hands_left + hands_right

    parser_mask_fixed = np.logical_or(parser_mask_fixed, parse_head)
    # ★ FIT HOOK: fewer passes = tight mask (slim), more = roomy mask (loose)
    parse_mask = cv2.dilate(parse_mask, np.ones((5, 5), np.uint16),
                            iterations=FIT_DILATE.get(fit, 5))
    if category == 'dresses' or category == 'upper_body':
        neck_mask = (parse_array == 18).astype(np.float32)
        neck_mask = cv2.dilate(neck_mask, np.ones((5, 5), np.uint16), iterations=1)
        neck_mask = np.logical_and(neck_mask, np.logical_not(parse_head))
        parse_mask = np.logical_or(parse_mask, neck_mask)
        arm_mask = cv2.dilate(np.logical_or(im_arms_left, im_arms_right).astype('float32'),
                              np.ones((5, 5), np.uint16), iterations=4)
        parse_mask += np.logical_or(parse_mask, arm_mask)

    parse_mask = np.logical_and(parser_mask_changeable, np.logical_not(parse_mask))

    parse_mask_total = np.logical_or(parse_mask, parser_mask_fixed)
    inpaint_mask = 1 - parse_mask_total
    img = np.where(inpaint_mask, 255, 0)
    dst = hole_fill(img.astype(np.uint8))
    dst = refine_mask(dst)
    inpaint_mask = dst / 255 * 1
    if fit == "loose":
        # one extra growth pass so long/loose garments get room
        inpaint_mask = cv2.dilate((inpaint_mask * 255).astype(np.uint8),
                                  np.ones((5, 5), np.uint8), iterations=2).astype(np.float32) / 255.0
    if category == "upper_body":
        # Mask must not run far below the original garment's hem (white-bar guard).
        _rows = np.where(np.isin(parse_array, (4, 7)))[0]
        if len(_rows):
            _cap = int(_rows.max() + FIT_BOTTOM_MARGIN.get(fit, 20))
            inpaint_mask[_cap:, :] = 0
    mask = Image.fromarray(inpaint_mask.astype(np.uint8) * 255)
    mask_gray = Image.fromarray(inpaint_mask.astype(np.uint8) * 127)

    return mask, mask_gray


# ATR/LIP label ids that are the PERSON (never the garment).
SKIN_LABELS = (1, 2, 3, 11, 12, 13, 14, 15, 18)  # hat, hair, sunglasses, head, legs, arms, neck


def restore_skin(orig_img: Image.Image, gen_img: Image.Image,
                 orig_parse: Image.Image, gen_parse: Image.Image) -> Image.Image:
    """Where BOTH parses say skin -> keep the ORIGINAL camera pixels (bit-exact)."""
    W, H = orig_img.size
    o = np.array(orig_parse.resize((W, H), Image.NEAREST))
    g = np.array(gen_parse.resize((W, H), Image.NEAREST))
    keep = np.isin(o, SKIN_LABELS) & np.isin(g, SKIN_LABELS)
    keep_f = cv2.GaussianBlur(keep.astype(np.float32), (0, 0), 1.0)[..., None]
    out = (np.array(orig_img).astype(np.float32) * keep_f
           + np.array(gen_img).astype(np.float32) * (1.0 - keep_f)).astype(np.uint8)
    return Image.fromarray(out)


def composite_final(orig_img: Image.Image, gen_img: Image.Image,
                    inpaint_mask: Image.Image, blur=2.0) -> Image.Image:
    """Outside the (feathered) garment mask -> ORIGINAL pixels, bit-exact."""
    W, H = orig_img.size
    m = np.array(inpaint_mask.resize((W, H), Image.BILINEAR)).astype(np.float32) / 255.0
    m = cv2.GaussianBlur(m, (0, 0), blur)[..., None]
    out = (np.array(orig_img).astype(np.float32) * (1.0 - m)
           + np.array(gen_img).astype(np.float32) * m).astype(np.uint8)
    return Image.fromarray(out)


GARMENT_LABELS = {"upper": (4, 7), "lower": (5, 6), "dress": (4, 5, 6, 7)}


def repair_hem(orig_img: Image.Image, gen_img: Image.Image,
               orig_parse: Image.Image, gen_parse: Image.Image,
               inpaint_mask: Image.Image, category="upper", feather=1.5) -> Image.Image:
    """Kill light leftover strips below the new garment's hem.

    Finds the bottom edge of the AI-painted garment (from the generated
    image's parse). Inside the repaint mask but BELOW that edge, wherever
    the model did not paint garment pixels (white/gray leftovers),
    restore the ORIGINAL photo pixels (usually the real jeans/legs).
    """
    W, H = orig_img.size
    g = np.array(gen_parse.resize((W, H), Image.NEAREST))
    labels = GARMENT_LABELS.get(category, (4, 7))
    grows = np.where(np.isin(g, labels))[0]
    if len(grows) == 0:
        return gen_img
    hem = int(np.percentile(grows, 99.5))  # robust hem line, ignores stray pixels
    m = np.array(inpaint_mask.resize((W, H), Image.NEAREST)) > 128
    below = np.zeros_like(m)
    below[hem:, :] = True
    leftover = m & below & (~np.isin(g, labels))
    if not leftover.any():
        return gen_img
    w = cv2.GaussianBlur(leftover.astype(np.float32), (0, 0), feather)[..., None]
    out = (np.array(orig_img).astype(np.float32) * w
           + np.array(gen_img).astype(np.float32) * (1.0 - w)).astype(np.uint8)
    return Image.fromarray(out)


# ---------------- self-test (runs without GPU) ----------------
if __name__ == "__main__":
    # Fake person parse: head + tee + arms + pants + legs on 384x512.
    P = np.zeros((512, 384), np.uint8)
    P[40:110, 162:222] = 11            # head
    P[150:330, 140:244] = 4            # upper clothes
    P[150:330, 100:140] = 14           # left arm
    P[150:330, 244:284] = 15           # right arm
    P[330:470, 140:244] = 6            # pants
    parse = Image.fromarray(P, mode="L")
    kp = {"pose_keypoints_2d": [[0, 0]] * 18}
    kp["pose_keypoints_2d"][2] = [150, 160]   # shoulder R
    kp["pose_keypoints_2d"][5] = [234, 160]   # shoulder L
    kp["pose_keypoints_2d"][3] = [140, 240]   # elbow R
    kp["pose_keypoints_2d"][6] = [244, 240]   # elbow L
    kp["pose_keypoints_2d"][4] = [135, 320]   # wrist R
    kp["pose_keypoints_2d"][7] = [249, 320]   # wrist L

    counts = {}
    for fit in ["slim", "regular", "loose"]:
        m, _ = get_mask_location("dc", "upper_body", parse, kp, fit=fit)
        counts[fit] = int((np.array(m) > 128).sum())
    print("mask pixels:", counts)
    assert counts["slim"] < counts["regular"] < counts["loose"], "fit must grow the mask!"
    assert counts["slim"] > 1000, "mask collapsed!"

    # Bit-exactness proofs (no randomness).
    rng = np.random.RandomState(0)
    orig = Image.fromarray(rng.randint(0, 255, (768, 576, 3), np.uint8))
    gen = Image.fromarray(rng.randint(0, 255, (768, 576, 3), np.uint8))
    skin_parse = Image.fromarray(np.full((512, 384), 11, np.uint8))  # all head
    assert np.array_equal(np.array(restore_skin(orig, gen, skin_parse, skin_parse)), np.array(orig))
    empty = Image.fromarray(np.zeros((512, 384), np.uint8))
    full = Image.fromarray(np.full((512, 384), 255, np.uint8))
    assert np.array_equal(np.array(composite_final(orig, gen, empty)), np.array(orig))
    assert np.array_equal(np.array(composite_final(orig, gen, full)), np.array(gen))
    # Mask-bottom cap: upper mask must not run far below the original tee.
    mcap = np.array(get_mask_location("dc", "upper_body", parse, kp, fit="regular")[0]) > 128
    teebot = int(np.where(P == 4)[0].max())
    assert np.where(mcap)[0].max() <= teebot + 20 + 1, "mask bottom cap failed!"
    print("mask bottom cap OK (tee ends at row %d, mask ends at row %d)" % (teebot, np.where(mcap)[0].max()))

    # Hem repair: white leftover strip below the AI hem must become original pixels.
    Wo, Ho = 288, 384
    oimg = Image.fromarray(np.full((Ho, Wo, 3), (30, 40, 90), np.uint8))     # dark jeans-ish
    gimg = Image.fromarray(np.full((Ho, Wo, 3), (200, 30, 30), np.uint8))    # red shirt-ish
    ga = np.array(gimg); ga[300:, :, :] = 255                                 # white strip below hem
    gimg = Image.fromarray(ga)
    oparse = Image.fromarray(np.full((Ho, Wo), 6, np.uint8))                  # orig: pants
    gparse_a = np.full((Ho, Wo), 0, np.uint8); gparse_a[:300, :] = 4           # gen: garment above, bg strip
    gparse = Image.fromarray(gparse_a)
    mhem = Image.fromarray(np.full((Ho, Wo), 255, np.uint8))                  # mask covers all
    fixed = np.array(repair_hem(oimg, gimg, oparse, gparse, mhem, "upper"))
    strip = fixed[320:, :, :].astype(np.float32)
    assert np.abs(strip - np.array((30, 40, 90), np.float32)).mean() < 12, "hem repair failed!"
    assert np.abs(np.array(gimg).astype(np.float32)[:280] - fixed[:280].astype(np.float32)).mean() < 1, "garment touched!"
    print("hem repair OK (white strip restored to original, garment untouched)")

    print("ALL MASK TESTS PASSED ✔ body preservation is bit-exact, fit grows slim < regular < loose")
