import os
import json
import torch
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_curve
import argparse

from scripts.detect_finetuned import FineTunedForensicDetector


class FineTunedEvaluator:
    """
    Evaluate fine-tuned ForensicHead on the held-out TEST split.
    Uses dataset_splits.json to ensure no data leakage.
    """
    def __init__(self, casia_dir, detector, split_file, device='cuda:0'):
        self.tp_dir = os.path.join(casia_dir, 'Tp')
        self.au_dir = os.path.join(casia_dir, 'Au')
        self.gt_dir = os.path.join(casia_dir, 'CASIA 2 Groundtruth')
        self.detector = detector
        self.device = device
        
        # Load test split
        with open(split_file, 'r') as f:
            splits = json.load(f)
        self.test_files = splits['test']
        
        # Authentic images (not in any split, used for image-level AUC)
        self.au_files = sorted([f for f in os.listdir(self.au_dir) 
                                if f.lower().endswith(('.jpg', '.png', '.tif', '.bmp'))])
        
        print(f"Test split: {len(self.test_files)} tampered images")
        print(f"Authentic images: {len(self.au_files)} available")
        
    def _get_gt_path(self, tp_filename):
        basename = os.path.splitext(tp_filename)[0]
        gt_name = basename + "_gt.png"
        gt_path = os.path.join(self.gt_dir, gt_name)
        if os.path.exists(gt_path):
            return gt_path
        return None

    def evaluate(self, num_samples=None):
        """
        Evaluate on the test split.
        If num_samples is None, evaluate on ALL test images.
        """
        if num_samples is None:
            num_samples = len(self.test_files)
        
        print(f"\nEvaluating on {min(num_samples, len(self.test_files))} test images...")
        
        all_pixel_preds = []
        all_pixel_gts = []
        image_scores = []
        image_labels = []
        
        # 1. Evaluate Tampered Images (from test split only)
        tp_count = 0
        for f in tqdm(self.test_files, desc="Tampered (Test)"):
            if tp_count >= num_samples:
                break
                
            gt_path = self._get_gt_path(f)
            if not gt_path:
                continue
                
            img_path = os.path.join(self.tp_dir, f)
            try:
                heatmap = self.detector.detect(img_path)
                
                gt_img = Image.open(gt_path).convert('L')
                gt_array = np.array(gt_img.resize((224, 224)))
                gt_binary = (gt_array > 128).astype(float)
                
                if gt_binary.sum() < 10:
                    continue
                
                pred_resized = cv2.resize(heatmap, (224, 224))
                
                all_pixel_preds.append(pred_resized.flatten())
                all_pixel_gts.append(gt_binary.flatten())
                image_scores.append(pred_resized.max())
                image_labels.append(1)
                
                tp_count += 1
            except Exception as e:
                print(f"Error processing {f}: {e}")
                continue

        # 2. Evaluate Authentic Images (same count as tampered for balance)
        au_count = 0
        for f in tqdm(self.au_files, desc="Authentic"):
            if au_count >= tp_count:
                break
            
            img_path = os.path.join(self.au_dir, f)
            try:
                heatmap = self.detector.detect(img_path)
                pred_resized = cv2.resize(heatmap, (224, 224))
                image_scores.append(pred_resized.max())
                image_labels.append(0)
                au_count += 1
            except Exception as e:
                print(f"Error processing {f}: {e}")
                continue

        # 3. Compute Metrics
        print("\nComputing metrics...")
        preds = np.concatenate(all_pixel_preds)
        gts = np.concatenate(all_pixel_gts)
        
        pixel_auc = roc_auc_score(gts, preds)
        precisions, recalls, thresholds = precision_recall_curve(gts, preds)
        f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
        best_f1 = f1_scores.max()
        
        image_auc = roc_auc_score(image_labels, image_scores)
        
        return {
            'pixel_auc': pixel_auc,
            'best_f1': best_f1,
            'image_auc': image_auc,
            'tp_count': tp_count,
            'au_count': au_count,
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--casia_dir', default='/home/uslib/quynhhuong/datasets/CASIA2/CASIA2')
    parser.add_argument('--checkpoint', default='/home/uslib/quynhhuong/ijepa/pretrained_models/IN1K-vit.h.14-300e.pth.tar')
    parser.add_argument('--head', default='/home/uslib/quynhhuong/ijepa/forensic_outputs/forensic_head_best.pt')
    parser.add_argument('--split_file', default='/home/uslib/quynhhuong/ijepa/forensic_outputs/dataset_splits.json')
    parser.add_argument('--num_samples', type=int, default=None, help='Number of test images (None=all)')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    
    detector = FineTunedForensicDetector(args.checkpoint, args.head, device=f'cuda:{args.gpu}')
    evaluator = FineTunedEvaluator(args.casia_dir, detector, args.split_file)
    
    results = evaluator.evaluate(num_samples=args.num_samples)
    
    print("\n" + "=" * 50)
    print("=== Evaluation Results (Fine-Tuned, TEST Split) ===")
    print("=" * 50)
    print(f"Pixel AUC:     {results['pixel_auc']:.4f}")
    print(f"Best Pixel F1: {results['best_f1']:.4f}")
    print(f"Image AUC:     {results['image_auc']:.4f}")
    print(f"Tampered:      {results['tp_count']} images")
    print(f"Authentic:     {results['au_count']} images")
