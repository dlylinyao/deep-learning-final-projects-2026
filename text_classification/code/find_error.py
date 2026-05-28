error_count = 0

with open("ground_truth.txt", "r") as f_gt, open(
    "my_predictions_512.txt", "r"
) as f_pred:
    gts = [line.strip().split() for line in f_gt.readlines()]
    preds = [line.strip().split() for line in f_pred.readlines()]

for i in range(len(gts)):
    gt_set = set(gts[i])
    pred_set = set(preds[i])

    if gt_set != pred_set:
        print(f"--- Line {i} ---")
        print(f"Ground Truth: {gt_set}")
        print(f"Prediction: {pred_set}")
        print("----------------")

        error_count += 1
        if error_count >= 5:
            break
