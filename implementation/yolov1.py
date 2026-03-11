import argparse

def main():
    parser = argparse.ArgumentParser(description='YOLOv1 - Train or Inference')
    parser.add_argument('--mode', choices=['train', 'infer', 'evaluate'], default='train',
                        help='Mode: train, infer (sample detections), or evaluate (mAP)')
    parser.add_argument('--config', dest='config_path', default='config/voc.yaml', type=str,
                        help='Path to config YAML file')
    args = parser.parse_args()

    if args.mode == 'train':
        from tools.train import train
        train(args)
    elif args.mode == 'infer':
        import torch
        from tools.inference import infer
        with torch.no_grad():
            infer(args)
    elif args.mode == 'evaluate':
        import torch
        from tools.inference import evaluate_map
        with torch.no_grad():
            evaluate_map(args)

if __name__ == '__main__':
    main()
